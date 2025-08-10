from diophantineequations.distributed.messages import WorkerToMaster, ModelType, MasterToWorker, \
    ActionNoSearch, WorkerResponse, ResponseNoSearch, WorkerAction, ResponseEnvironment, ActionTactics, \
    ResponseEnvironmentWholeProof

from typing import Optional
import logging
from time import time_ns
from diophantineequations.notify import send_notification
from diophantineequations.distributed.workers.search.abstract import AbstractSearchWorker
from diophantineequations.utils import replace_sorry

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class NoSearchWorker(AbstractSearchWorker):
    def __init__(self, trainer_exchange: str, db_path: str, initial_model_path: str, initial_model_type: ModelType):
        super().__init__(trainer_exchange, db_path, initial_model_path, initial_model_type)
        self._tactics = []

    def do_work(self) -> None:
        assert isinstance(self._action, ActionNoSearch)
        # Reset search
        self._tactics = []
        goal = replace_sorry(self._theorem_statement, "")
        # Send to tactic expansion
        msg = MasterToWorker(message=WorkerAction(action=ActionTactics(search_idx=self._action.search_idx,
                                                                       goal=goal,
                                                                       premises=self._action.premises,
                                                                       k=self._action.num_tactics,
                                                                       model_type=self.model_type,
                                                                       model_path=self.model_path,
                                                                       theorem=self._theorem_statement,
                                                                       past_tactics=[],
                                                                       run_config=self._action.run_config)))
        self._send_to_tactic(msg)
        self._awaiting_env += 1
        self._awaiting_tactics += 1

    def handle_done(self, proven: bool, diff: int, error: Optional[str]) -> None:
        search_time_end = time_ns() // 1000 // 1000
        logger.info("Handling done, proven: %s, error: %s", proven, error)
        assert isinstance(self._action, ActionNoSearch)
        if proven:
            assert error is None
            proof = "\n".join(self._tactics)
            logger.info("Proven with proof %s", proof)
            msg = WorkerToMaster(message=WorkerResponse(
                response=ResponseNoSearch(action=self._action, proof=proof, error=None, ms_between=diff,
                                          search_time_ms=search_time_end - self._search_start_time,
                                          depth=len(self._tactics))))
            self._send_response_to_root(msg)
            return

        msg = WorkerToMaster(message=WorkerResponse(
            response=ResponseNoSearch(action=self._action, proof=None, error=error, ms_between=diff,
                                      search_time_ms=search_time_end - self._search_start_time,
                                      depth=len(self._tactics))))
        self._send_response_to_root(msg)

    def handle_whole_proof(self, response: ResponseEnvironmentWholeProof, diff: int) -> None:
        self._awaiting_env -= 1
        for idx, (proven, error) in enumerate(zip(response.proven, response.errors, strict=True)):
            assert not proven or error is None
            if proven:
                self._tactics.append(response.action.proofs[idx])
                self.handle_done(proven=proven, diff=diff, error=error)
                return
        self.handle_done(False, diff, "Not a single proof worked!")

    def env_callback(self, response: WorkerResponse) -> None:
        curr = time_ns() // 1000 // 1000
        diff = curr - self._last_action
        response = response.response
        assert isinstance(self._action, ActionNoSearch)
        assert isinstance(response, ResponseEnvironment) or isinstance(response, ResponseEnvironmentWholeProof)
        if response.ms_between > 3000:
            logger.warning("Env waited more than 3 seconds, consider changing the worker distribution!")
        if isinstance(response, ResponseEnvironmentWholeProof):
            self.handle_whole_proof(response=response, diff=diff)
            return

        # Create env expansion and expand
        if response.error:
            self.handle_done(False, diff, response.error)
            self._awaiting_env -= 1
            return

        for idx, proof_state in enumerate(response.next_proof_states):
            if proof_state is None:
                continue
            if proof_state.proof_status == "Completed":
                self._tactics.append(response.action.current_tactics[idx])
                self.handle_done(True, diff, response.error)
                self._awaiting_env -= 1
                return
        # If not proven, enqueue if not already at max depth
        if len(self._tactics) >= self._action.max_tactics:
            self.handle_done(False, diff, "Max depth reached")
            self._awaiting_env -= 1
            return
        for idx, proof_state in enumerate(response.next_proof_states):
            if proof_state is not None and proof_state.lean_code_is_valid(allow_sorry=False):
                self._tactics.append(response.action.current_tactics[idx])
                goal = replace_sorry(self._theorem_statement, "\n".join(self._tactics))
                msg = MasterToWorker(message=WorkerAction(action=ActionTactics(search_idx=self._action.search_idx,
                                                                               goal=goal,
                                                                               premises=self._action.premises,
                                                                               k=self._action.num_tactics,
                                                                               model_type=self.model_type,
                                                                               model_path=self.model_path,
                                                                               theorem=self._theorem_statement,
                                                                               past_tactics=self._tactics,
                                                                               run_config=self._action.run_config)))
                self._send_to_tactic(msg)
                self._awaiting_tactics += 1
                return
        # None was valid
        self.handle_done(False, diff, "No valid tactic!")
        self._awaiting_env -= 1
        assert self._awaiting_env >= 0


def main():
    worker = NoSearchWorker.from_env_vars()
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "NoSearch")
        raise e
    send_notification(False, job_name="NoSearch")
