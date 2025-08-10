import regex
from diophantineequations.distributed.workers.abstract import Worker, RecoverableError
from diophantineequations.distributed.messages import MasterToWorker, ActionEnvironment, \
    ResponseEnvironment, WorkerToMaster, WorkerResponse, ActionInitialEnvironment, ResponseInitialEnvironment, \
    ActionEnvironmentAddHypotheses, ResponseEnvironmentAddHypotheses, ActionEnvironmentWholeProof, \
    ResponseEnvironmentWholeProof
from pathlib import Path
from lean_interact import AutoLeanServer, Command, LeanREPLConfig, LocalProject
from lean_interact.server import LeanError
from lean_interact.interface import Sorry, ProofStep, ProofStepResponse, GetDeclType, CommandResponse
from diophantineequations.notify import send_notification
from diophantineequations.utils import _find_leftmost_not_in_parenthesis, text_without_comments, replace_sorry
from typing import Optional, Tuple
import logging
from time import time_ns
from functools import partial
import threading

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

HYP_REGEX = r'''\(\s*(?P<name>[^():]+?)\s*:\s*(?P<type>(?:[^()]++|\((?:(?P>type)|[^()]++)*\))*)\)'''


class EnvWorker(Worker):
    def __init__(self, lean_path: Path, lean_header: str):
        # Lean setup, done before we start the connection, as we block the thread
        self.lean_path = lean_path
        self._lean_header = lean_header
        self._env_config = LeanREPLConfig(project=LocalProject(str(lean_path.resolve()), build=False), memory_hard_limit_mb=12_500,
                                          repl_rev="v1.0.9", repl_git="https://github.com/sorgfresser/repl")
        self.handler: AutoLeanServer
        self._env: int
        self._reset_handler(full=True)
        # Now do the connection setup
        super().__init__(prefetch_count=1)
        self.queues.environment.declare(self.channel)
        self.queues.environment.consume(self.channel, self.callback)
        self._threaded = partial(self.threaded_work, work=self.work, send_reply_factory=self.send_reply)
        self._subsequent_failures = 0
        self._last_action = time_ns() // 1000 // 1000  # time in milliseconds

    def get_callback(self):
        return self._callback

    def _reset_handler(self, full: bool = False):
        if full:
            self.handler: AutoLeanServer = AutoLeanServer(config=self._env_config, max_total_memory=0.95,
                                                          max_process_memory=0.95)
            result = self.handler.run(Command(cmd=self._lean_header), add_to_session_cache=True)
            if isinstance(result, LeanError):
                raise RuntimeError("Could not load header")
            self._env = result.env
        else:
            self.handler.restart()

    def send_reply(self, ch, method, properties, msg: WorkerToMaster):
        def send_response():
            logger.debug("Sending response %s", msg.model_dump_json())
            ch.basic_publish(exchange="", routing_key=properties.reply_to, body=msg.model_dump_json())
            logger.info("Completed message!")
            ch.basic_ack(delivery_tag=method.delivery_tag)

        return send_response

    def _callback(self, ch, method, properties, body):
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        logger.info("Received message!")
        if not isinstance(msg.action, ActionEnvironment) and not isinstance(msg.action,
                                                                            ActionInitialEnvironment) and not isinstance(
            msg.action, ActionEnvironmentAddHypotheses) and not isinstance(msg.action, ActionEnvironmentWholeProof):
            raise RuntimeError("Action received was not an ActionEnvironment, might be the wrong queue!")
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        thread_fn = partial(self._threaded, ch, method, properties, action=msg.action)
        thread = threading.Thread(target=thread_fn)
        thread.start()

    def work(self,
             action: ActionInitialEnvironment | ActionEnvironment | ActionEnvironmentAddHypotheses | ActionEnvironmentWholeProof) -> WorkerResponse:
        curr = time_ns() // 1000 // 1000
        diff = curr - self._last_action
        if isinstance(action, ActionEnvironment):
            work = self.do_work
        elif isinstance(action, ActionEnvironmentAddHypotheses):
            work = self.inject_hypothesis
        elif isinstance(action, ActionEnvironmentWholeProof):
            work = self.whole_proof
        else:
            work = self.do_initial_work
        try:
            response = work(action, diff)
        except (ConnectionAbortedError) as e:
            logger.exception("Connection aborted")
            self._subsequent_failures += 1
            if self._subsequent_failures > 3:
                self._reset_handler(full=True)
                raise RuntimeError("Too many failures!") from e
            self._reset_handler()
            raise RecoverableError from e
        self._subsequent_failures = 0
        return WorkerResponse(response=response)

    def _get_proof_state(self, theorem: str, past_tactics: list[str], context: Optional[str]) -> Tuple[
        Optional[Sorry | ProofStepResponse], Optional[str]]:
        env = self.get_env(context)
        # This one can time out if the formalization is poor
        try:
            response = self.handler.run(Command(cmd=theorem, env=env), timeout=150)
        except TimeoutError:
            error = "Formalization timed out!"
            logger.exception(error)
            self._reset_handler()
            return None, error
        except ConnectionAbortedError:
            error = "Connection aborted!"
            logger.exception(error)
            self._reset_handler()
            return None, error
        if isinstance(response, LeanError):
            error = response.message
            logger.info("Had error %s", error)
            return None, error
        if not response.lean_code_is_valid(allow_sorry=True):
            errors: list[str] = [msg.data for msg in response.get_errors()]
            logger.info("Had errors %s", errors)
            return None, "\n".join(errors)
        sorries = response.sorries
        has_exactly_one_sorry = len(sorries) == 1
        logger.debug("Received theorem %s, has_exactly_one_sorry: %s", theorem, has_exactly_one_sorry)
        if not has_exactly_one_sorry:
            logger.warning("Theorem %s had not exactly one sorry", theorem)
            error = f"Had {len(sorries)} sorries instead of 1!"
            return None, error
        response = sorries[0]
        logger.debug("Received one sorry %s", response)
        proof_state_idx = response.proof_state
        # Apply past tactics
        for tactic in past_tactics:
            response = self.handler.run(ProofStep(proof_state=proof_state_idx, tactic=tactic.strip()), timeout=240)
            if isinstance(response, LeanError):
                error = response.message
                logger.error("Error when applying past tactics: %s", error)
                return None, error
            assert isinstance(response, ProofStepResponse)
            if not response.lean_code_is_valid(allow_sorry=False):
                errors: list[str] = [msg.data for msg in response.get_errors()]
                logger.error("Error when applying past tactics: %s", errors)
                return None, "\n".join(errors)
            proof_state_idx = response.proof_state
        return response, None

    def do_initial_work(self, action: ActionInitialEnvironment, diff: int) -> ResponseInitialEnvironment:
        state, error = self._get_proof_state(action.theorem, past_tactics=action.past_tactics, context=action.context)
        return ResponseInitialEnvironment(action=action, proof_state=state, error=error, ms_between=diff)

    def inject_hypothesis(self, action: ActionEnvironmentAddHypotheses, diff: int):
        env = self.get_env(action.context)
        lean_types = []
        theorem_part: str = text_without_comments(action.theorem)
        if not "theorem" in theorem_part and not "lemma" in theorem_part:
            logger.error("Theorem part is missing: %s", theorem_part)
            return ResponseEnvironmentAddHypotheses(action=action, theorem=theorem_part, ms_between=diff, error="No theorem or lemma found!")

        for decl in action.decls:
            # Remove imports
            decl = text_without_comments(decl)
            line_idx = 0
            for idx, line in enumerate(decl.splitlines()):
                if line.startswith("import") or not line.strip():
                    continue
                line_idx = idx
                break
            decl = "\n".join(decl.splitlines()[line_idx:])
            try:
                response = self.handler.run(GetDeclType(decl=decl, env=env), timeout=300)
            except (TimeoutError, ConnectionAbortedError) as e:
                logger.info("Decl %s raised error %s", decl, e)
                self._reset_handler()
                env = self.get_env(action.context)
                continue
            if isinstance(response, LeanError):
                logger.error("Error when getting type for decl: %s", response)
                logger.info("Decl was: %s", decl)
                continue
            if not response.lean_code_is_valid(allow_sorry=False):
                logger.error("Error when getting type for decl: %s", response)
                logger.info("Decl was: %s", decl)
                continue
            if not response.types:
                logger.error("No type returned!")
                logger.info("Decl was: %s", decl)
                continue
            lean_types.append(response.types[-1])

        split_word = "theorem" if "theorem" in theorem_part else "lemma"
        before, theorem = theorem_part.split(split_word, maxsplit=1)
        theorem_statement = theorem[:_find_leftmost_not_in_parenthesis(theorem, ":=")]
        hyp_matches = regex.findall(HYP_REGEX, theorem_statement)
        hyp_names = [match[0].strip() for match in hyp_matches]
        skip_mask = [False] * len(lean_types)
        if not lean_types:
            return ResponseEnvironmentAddHypotheses(action=action, theorem=theorem_part, ms_between=diff, error=None)
        while not all(skip_mask):
            hypotheses = []
            idx = 0
            for type_idx, lean_type in enumerate(lean_types):
                if skip_mask[type_idx]:
                    continue
                while f"h{idx}" in hyp_names:
                    idx += 1
                hypotheses.append(f"(h{idx} : {lean_type})")
                idx += 1
            theorem_statement = theorem[:_find_leftmost_not_in_parenthesis(theorem, ":=")]
            split = _find_leftmost_not_in_parenthesis(theorem_statement, ":")
            hyp_part, conclusion = theorem_statement[:split], theorem_statement[split:]
            hyp_part += " ".join(hypotheses)
            theorem_statement = hyp_part + " " + conclusion
            new_theorem = theorem_statement + theorem[_find_leftmost_not_in_parenthesis(theorem, ":="):]
            full = split_word.join([before, new_theorem])
            # Check that it actually works
            response = None
            try:
                response = self.handler.run(Command(cmd=full, env=env), timeout=300)
            except (TimeoutError, ConnectionAbortedError) as e:
                logger.info("Theorem with decls %s raised error %s", full, e)
                self._reset_handler()
            if isinstance(response, LeanError):
                logger.error("Error when trying constructed theorem: %s", response.message)
            elif response is not None:
                assert isinstance(response, CommandResponse)
                if not response.lean_code_is_valid(allow_sorry=True):
                    errors: list[str] = [msg.data for msg in response.get_errors()]
                    logger.error("Error when trying constructed theorem: %s", errors)
                else: # Valid
                    break
            # Set another type to skip
            first_valid = next((i for i, skip in enumerate(skip_mask) if not skip), None)
            if first_valid is not None:
                skip_mask[first_valid] = True
        if response:
            return ResponseEnvironmentAddHypotheses(action=action, theorem=full, ms_between=diff, error=None)
        return ResponseEnvironmentAddHypotheses(action=action, theorem=theorem_part, ms_between=diff,error="Could not construct a valid theorem with the given hypotheses!")

    def get_env(self, context: Optional[str]) -> int:
        if context:
            response = self.handler.run(Command(cmd=context, env=self._env), timeout=300)
            logger.debug("Context response: %s", response)
            env = response.env
        else:
            logger.debug("No context set!")
            env = self._env
        return env

    def whole_proof(self, action: ActionEnvironmentWholeProof, diff: int) -> ResponseEnvironmentWholeProof:
        env = self.get_env(action.context)
        proven_list = []
        error_list = []
        for proof in action.proofs:
            error = None
            theorem = replace_sorry(action.theorem, proof)
            try:
                response = self.handler.run(Command(cmd=theorem, env=env), timeout=15)
            except (TimeoutError, ConnectionAbortedError) as e:
                logger.info("Theorem %s raised error %s", theorem, e)
                self._reset_handler()
                env = self.get_env(action.context)
                response = None
                error = str(e)
            if response is None:
                proven = False
            else:
                if isinstance(response, LeanError):
                    proven = False
                    error = response.message
                else:
                    assert isinstance(response, CommandResponse)
                    if not response.lean_code_is_valid(allow_sorry=False):
                        proven = False
                        errors: list[str] = [msg.data for msg in response.get_errors()]
                        error = "\n".join(errors)
                    else:
                        proven = True
            proven_list.append(proven)
            error_list.append(error)
            logger.debug("Proven: %s", proven)
            logger.debug("Lean code: %s", theorem)
            logger.debug("Error: %s", error)
        return ResponseEnvironmentWholeProof(action=action, ms_between=diff, proven=proven_list, errors=error_list)

    def do_work(self, action: ActionEnvironment, diff: int) -> ResponseEnvironment:
        def get_state():
            s, e = self._get_proof_state(action.theorem, past_tactics=action.past_tactics, context=action.context)
            if s is None:
                logger.error("Error when trying to get the initial proof state for theorem!")
                logger.error("Theorem: %s", action.theorem)
                logger.error("Past tactics: %s", action.past_tactics)
                logger.error("Error: %s", e)
            return s, e

        state, error = get_state()
        if state is None:
            return ResponseEnvironment(action=action, error=error, next_proof_states=[], ms_between=diff)
        proof_state_idx = state.proof_state
        if not action.current_tactics:
            logger.warning("Empty tactics received!")
            return ResponseEnvironment(action=action, error="Empty tactics!", next_proof_states=[], ms_between=diff)

        # Apply new tactics
        new_proof_states = []
        for tactic in action.current_tactics:
            tactic = tactic.strip()
            if not tactic:
                logger.debug("Empty tactic!")
                new_proof_states.append(None)
                continue
            try:
                response = self.handler.run(ProofStep(proof_state=proof_state_idx, tactic=tactic), timeout=30)
            except (TimeoutError, ConnectionAbortedError) as e:
                logger.info("Tactic %s raised error %s", tactic, e)
                self._reset_handler()
                state, error = get_state()
                if state is None:
                    return ResponseEnvironment(action=action, error=error, next_proof_states=[], ms_between=diff)
                proof_state_idx = state.proof_state
                response = None
            if response is not None and isinstance(response, LeanError):
                logger.debug("Error on tactic: %s\nError: %s", tactic, response)
                response = None
            if response is not None and not response.lean_code_is_valid(allow_sorry=False):
                logger.debug("Error on tactic: %s\nError: %s", tactic, response)
                response = None
            new_proof_states.append(response)

        return ResponseEnvironment(action=action, error=None, next_proof_states=new_proof_states, ms_between=diff)


def main():
    from os import getenv
    lean_path = getenv("LEAN_PATH")
    if lean_path is None:
        raise ValueError("Lean path not set!")
    lean_header = getenv("LEAN_HEADER")
    if lean_header is None:
        raise ValueError("Lean header not set!")
    lean_header = lean_header.replace("\\n", "\n")
    lean_header = lean_header.removeprefix("\"").removesuffix("\"")
    lean_path = Path(lean_path)
    if not lean_path.exists():
        raise ValueError("Lean path does not exist!")
    worker = EnvWorker(lean_path, lean_header)
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Environment")
        raise e
    send_notification(False, job_name="Environment")
