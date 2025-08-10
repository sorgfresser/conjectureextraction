import math

from lean_interact.interface import ProofStepResponse, Sorry
from diophantineequations.distributed.workers.search.abstract import AbstractSearchWorker
from diophantineequations.distributed.messages import WorkerToMaster, ModelType, MasterToWorker, \
    ActionSearch, WorkerResponse, ResponseSearch, ActionInitialEnvironment, WorkerAction, \
    ResponseInitialEnvironment, ResponseEnvironment, ActionTactics, ActionTrainSample, ResponseTactics, \
    ActionEnvironment
from diophantineequations.utils import replace_sorry
from htps import Context, SearchParams, PolicyType, QValueSolved, NodeMask, Metric, HTPS, Theorem, EnvExpansion, \
    SampleTactics, Proof, EnvEffect, Tactic, Result
from typing import Optional, Dict
from sqlmodel import SQLModel, Field, select, Session, Column, JSON, PrimaryKeyConstraint, func
import json
import logging
from time import time_ns
from diophantineequations.notify import send_notification
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def get_proof(proof: Proof) -> list[str]:
    tactics = [proof.tactic.unique_string]
    if not proof.children:
        return tactics
    return tactics + get_proof(proof.children[0])


class Search(SQLModel, table=True):
    search: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    search_idx: int = Field(primary_key=True)
    num_expansions: int = Field(primary_key=True)
    trial: int = Field(primary_key=True)
    run_config: Dict = Field(default_factory=dict, sa_column=Column(JSON))

    __table_args__ = (PrimaryKeyConstraint("search_idx", "num_expansions", "trial"),)


class Expansion(SQLModel, table=True):
    expansion: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    search_idx: int = Field(primary_key=True)
    num_expansions: int = Field(primary_key=True)
    trial: int = Field(primary_key=True)  # increasing number to avoid uniqueness errors in case of rescheduled searches
    run_config: Dict = Field(default_factory=dict, sa_column=Column(JSON))

    __table_args__ = (PrimaryKeyConstraint("search_idx", "num_expansions", "trial"),)


class SearchWorker(AbstractSearchWorker):
    def __init__(self, trainer_exchange: str, db_path: str, initial_model_path: str, initial_model_type: ModelType):
        super().__init__(trainer_exchange, db_path, initial_model_path, initial_model_type)
        self._search = None

    def _get_trial(self, search_idx: int) -> int:
        with Session(self.engine) as session:
            stmt = select(func.max(Search.trial)).where(Search.search_idx == search_idx)
            version = session.exec(stmt).first()
        if version is None:
            version = 0
        else:
            version += 1
        return version

    def softmax(self, priors: list[float]) -> list[float]:
        p_exp = [math.exp(p) for p in priors]
        p_sum = sum(p_exp)
        p_softmaxed = [p / p_sum for p in p_exp]
        return p_softmaxed

    def _store(self, num_expansion: int, search: Optional[str] = None, expansion: Optional[str] = None):
        if not search and not expansion:
            raise ValueError("Neither search nor expansion set!")
        if search:
            with Session(self.engine) as session:
                session.add(
                    Search(search=json.loads(search), search_idx=self._action.search_idx, num_expansions=num_expansion,
                           trial=self._trial, run_config=self._action.run_config.model_dump(mode="json")))
                session.commit()
        if expansion:
            with Session(self.engine) as session:
                session.add(Expansion(expansion=json.loads(expansion), search_idx=self._action.search_idx,
                                      num_expansions=num_expansion, trial=self._trial,
                                      run_config=self._action.run_config.model_dump(mode="json")))
                session.commit()

    def _delete_search(self):
        with Session(self.engine) as session:
            models = [Search, Expansion]
            for model in models:
                stmt = select(model).where(model.search_idx == self._action.search_idx)
                samples = session.exec(stmt)
                for sample in samples:
                    session.delete(sample)
                session.commit()
                assert session.exec(stmt).first() is None

    def tactics_callback(self, response: WorkerResponse) -> None:
        # Override tactics callback, as we never want to use whole proof generation
        tactics_response = response.response
        assert isinstance(tactics_response, ResponseTactics)
        # Cut down to only the first \n
        logger.debug("Before split: %s", tactics_response.strings)
        without_by = [re.sub(r"^\s*by\s*\n", "", string) for string in tactics_response.strings]
        cropped_strings = [string.strip("\n").split("\n")[0] for string in without_by]
        tactics_response.strings = cropped_strings
        if tactics_response.ms_between > 3000:
            logger.warning("Tactic generation waited more than 3 seconds, consider changing the worker distribution!")
        context = "\n".join(tactics_response.action.premises) if tactics_response.action.premises else None
        logger.debug("Theorem %s with context: %s", self._theorem_statement, context)
        logger.debug("Tactics: %s", tactics_response.strings)
        action = ActionEnvironment(
            theorem=self._theorem_statement, past_tactics=tactics_response.action.past_tactics, goal=tactics_response.action.goal,
            current_tactics=tactics_response.strings, current_logprobs=tactics_response.logprobs, search_idx=self._action.search_idx,
            run_config=self._action.run_config, context=context)
        msg = MasterToWorker(message=WorkerAction(action=action))
        self._send_to_environment(msg)

    def env_callback(self, response: WorkerResponse):
        response = response.response
        assert isinstance(response, ResponseInitialEnvironment) or isinstance(response, ResponseEnvironment)
        assert isinstance(self._action, ActionSearch)
        if response.ms_between > 3000:
            logger.warning("Env waited more than 3 seconds, consider changing the worker distribution!")
        curr = time_ns() // 1000 // 1000
        diff = curr - self._last_action
        if isinstance(response, ResponseInitialEnvironment):
            if response.proof_state is None:
                search_time = time_ns() - self._search_start_time
                search_time_ms = search_time // 1000 // 1000
                msg = WorkerToMaster(message=WorkerResponse(
                    response=ResponseSearch(action=self._action, proof=None, error=response.error, ms_between=diff,
                                            expansions=0, search_time_ms=search_time_ms)))
                self._send_response_to_root(msg)
                return
            assert isinstance(response.proof_state, Sorry)
            self._goal = response.proof_state.goal
            self._initial_search()
            return
        # Create env expansion and expand
        assert isinstance(response, ResponseEnvironment)
        goal = response.action.goal
        context = Context([])
        past_tactics = [Tactic(tac, True, 1) for tac in response.action.past_tactics]
        thm = Theorem(goal, goal, [], context, past_tactics)
        if response.error:
            expansion = EnvExpansion(thm, 1, 1, [], response.error)
        else:
            # If we have no next proof state, the tactic was erroneus. If we have no goals, but are not finished,
            # there are dangling metavars which we cannot handle.
            valid_ids = [i for i in range(len(response.next_proof_states)) if response.next_proof_states[i] is not None \
                         and (response.next_proof_states[i].goals or response.next_proof_states[i].proof_status == "Completed")]
            if not valid_ids:
                expansion = EnvExpansion(thm, 1, 1, [], "No valid ids!")
            else:
                tactics = [Tactic(response.action.current_tactics[i], True, 1) for i in valid_ids]
                proof_states: list[ProofStepResponse] = [response.next_proof_states[i] for i in valid_ids]
                if self.model_type == ModelType.deepseek:
                    children = [
                        [Theorem(goal + "\n" + tactic.unique_string, goal + "\n" + tactic.unique_string, [], context,
                                 past_tactics + [tactic])] if proof_state.goals else [] for proof_state, tactic in
                        zip(proof_states, tactics, strict=True)]
                else:
                    children = [
                        [Theorem(goal, goal, [], context, past_tactics + [tactic]) for goal in proof_state.goals] for
                        proof_state, tactic in zip(proof_states, tactics, strict=True)]
                # If we solve, we must only send solved
                solving_ids = [i for i in range(len(proof_states)) if proof_states[i].proof_status == "Completed"]
                if solving_ids:
                    tactics = [tactics[i] for i in solving_ids]
                    proof_states = [proof_states[i] for i in solving_ids]
                    children = [children[i] for i in solving_ids]
                effects = [EnvEffect(thm, tactic, children) for tactic, proof_state, children in
                           zip(tactics, proof_states, children, strict=True)]
                priors = response.action.current_logprobs
                priors = [priors[i] for i in valid_ids]
                if solving_ids:
                    priors = [priors[i] for i in solving_ids]
                priors = self.softmax(priors)
                expansion = EnvExpansion(thm, 1, 1, [0] * len(tactics), effects, -0.5, tactics, children, priors)
        self._store(self._search.expansions, expansion=expansion.get_json_str())
        self._search.expand_and_backup([expansion])
        self._awaiting_env -= 1
        assert self._awaiting_env >= 0
        if self._awaiting_env == 0:
            # Tactic gen
            theorems: list[Theorem] = self._search.theorems_to_expand()
            self._store(self._search.expansions, search=self._search.get_json_str())
            # Handle done
            if self._search.is_done():
                self.handle_done(self._search.proven(), 0, None)
            else:
                self._handle_theorems(theorems)

    def _initial_search(self):
        context = Context([])
        root_thm = Theorem(self._goal, unique_string=self._goal, hypotheses=[], context=context, past_tactics=[])
        params = SearchParams(0.3, PolicyType.RPO, num_expansions=self._action.num_expansions, succ_expansions=3,
                              early_stopping=False, no_critic=False, backup_once=False, backup_one_for_solved=True,
                              depth_penalty=0.99, count_threshold=10, tactic_p_threshold=True,
                              tactic_sample_q_conditioning=False, only_learn_best_tactics=True, tactic_init_value=0.0,
                              q_value_solved=QValueSolved.One, policy_temperature=0.7, metric=Metric.Size,
                              node_mask=NodeMask.Solving, effect_subsampling_rate=1.0,
                              critic_subsampling_rate=1.0,
                              early_stopping_solved_if_root_not_proven=True, virtual_loss=0)
        self._search = HTPS(root_thm, params)
        self._store(0, self._search.get_json_str())
        theorems: list[Theorem] = self._search.theorems_to_expand()
        self._handle_theorems(theorems)

    def handle_done(self, proven: bool, diff: int, error: Optional[str]) -> None:
        logger.info("Starting handle done!")
        curr = time_ns() // 1000 // 1000
        diff = curr - self._last_action
        assert isinstance(self._action, ActionSearch)
        result: Result = self._search.get_result() if self._search else None
        if result is None or not result.proof:
            proof = None
        else:
            proof_list = get_proof(result.proof)
            proof = "\n".join(proof_list)
        logger.info("Proof %s for theorem %s", proof, result.goal.conclusion if result is not None else "None")
        tactic_samples: list[SampleTactics] = result.tactic_samples if result is not None else []
        for sample in tactic_samples:
            # Choose tactic with the highest prob
            best_tactic_id = max(range(0, len(sample.tactics)), key=lambda x: sample.target_pi[x])
            msg = MasterToWorker(message=WorkerAction(action=ActionTrainSample(model_path=self.model_path,
                                                                               model_type=self.model_type,
                                                                               premises=self._action.premises,
                                                                               state=sample.goal.conclusion,
                                                                               tactic=sample.tactics[
                                                                                   best_tactic_id].unique_string,
                                                                               search_idx=self._action.search_idx,
                                                                               run_config=self._action.run_config)))
            self._send_to_trainer(msg)
        search_time = time_ns() - self._search_start_time
        search_time_ms = search_time // 1000 // 1000
        expansions = self._search.expansions if self._search is not None else 0
        msg = WorkerToMaster(message=WorkerResponse(
            response=ResponseSearch(action=self._action, proof=proof, error=None, ms_between=diff,
                                    expansions=expansions, search_time_ms=search_time_ms)))
        self._delete_search()
        self._send_response_to_root(msg)

    def _handle_theorems(self, theorems: list[Theorem]):
        self._awaiting_tactics += len(theorems)
        # need to set this early, as the order of tactics callback and env callback is unclear
        # so we might end up with 0 awaiting env though we did not process all tactics callback yet otherwise
        self._awaiting_env += len(theorems)
        for theorem in theorems:
            msg = MasterToWorker(message=WorkerAction(action=ActionTactics(search_idx=self._action.search_idx,
                                                                           goal=theorem.conclusion,
                                                                           premises=self._action.premises,
                                                                           k=self._action.num_tactics,
                                                                           model_type=self.model_type,
                                                                           model_path=self.model_path,
                                                                           theorem=self._theorem_statement,
                                                                           past_tactics=[tactic.unique_string for tactic
                                                                                         in theorem.past_tactics],
                                                                           run_config=self._action.run_config)))
            self._send_to_tactic(msg)

    def do_work(self) -> None:
        assert isinstance(self._action, ActionSearch)
        self._trial = self._get_trial(self._action.search_idx)
        # Use proof state instead of theorem statement
        if self.model_type == ModelType.reprover:
            msg = MasterToWorker(message=WorkerAction(
                action=ActionInitialEnvironment(theorem=self._theorem_statement, search_idx=self._action.search_idx,
                                                run_config=self._action.run_config, past_tactics=[], context=None)))
            self._send_to_environment(msg)
            return
        # Crop the by sorry part
        if self.model_type == ModelType.deepseek:
            self._goal = replace_sorry(self._theorem_statement, "")
        else:
            raise ValueError("Unknown model type")
        self._initial_search()

    def _search_callback(self, ch, method, properties, body):
        self._search = None
        super()._search_callback(ch, method, properties, body)


def main():
    worker = SearchWorker.from_env_vars()
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Search")
        raise e
    send_notification(False, job_name="Search")
