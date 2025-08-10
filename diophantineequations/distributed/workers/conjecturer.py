import contextlib

from diophantineequations.distributed.workers.abstract import Worker
from diophantineequations.distributed.messages import MasterToWorker, ActionConjecture, \
    WorkerToMaster, WorkerResponse, Conjecture, ResponseConjecture, WorkerAction, ActionFormalization, \
    ResponseFormalization
from diophantineequations.lemma_sketch import sketch_proof
import logging
from diophantineequations.lemma_gen import generate_conjectures
from typing import Optional, List
from diophantineequations.lemma_form import LeanFile
from diophantineequations.notify import send_notification
from time import time_ns
from sqlmodel import SQLModel, Field, create_engine, Session, Column, JSON
from filelock import Timeout, FileLock

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


class SQLConjecture(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    informal_problem: str
    parent_problem: Optional[str]
    formal_problem: Optional[str]
    definitions: List[str] = Field(sa_column=Column(JSON), default_factory=list)
    attempts: int


class ConjectureWorker(Worker):
    def __init__(self, model: str, db_path: str, num_formalizations: int = 10):
        super().__init__(prefetch_count=1)
        self.queues.conjecture.declare(self.channel)
        self.consumer_tag = self.queues.conjecture.consume(self.channel, self.callback)
        # Db
        if not db_path.startswith("sqlite:///"):
            db_path = f"sqlite:///{db_path}"
        self.lock = FileLock(f"{db_path}.lock")
        self.db_path = db_path
        self.engine = create_engine(db_path)
        with self.db_lock():
            SQLModel.metadata.create_all(self.engine)
        # Internals
        self._model = model
        self._conjectures: list[Conjecture] = []
        self._attempts = []
        self._done = []
        self._definitions: List[Optional[List[LeanFile]]] = []
        self._reply_to = None
        self._tag = None
        self._action: Optional[ActionConjecture] = None
        self._num_formalizations = num_formalizations
        # Formalizer queue
        self.formalizer_reply_to = self.get_exclusive(self._formalize_callback)
        self.queues.formalization.declare(self.channel)
        self._last_action = time_ns() // 1000 // 1000  # time in milliseconds

    @contextlib.contextmanager
    def db_lock(self):
        try:
            with self.lock.acquire(timeout=3600):
                yield self
        except Timeout:
            send_notification(True, "Timed out after 60 minutes!", "Conjecturer")

    def get_callback(self):
        return self._callback

    def _callback(self, ch, method, properties, body):
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        logger.info("Received message!")
        if not isinstance(msg.action, ActionConjecture):
            raise RuntimeError("Action received was not an ActionConjecture, might be the wrong queue!")
        self._reply_to = properties.reply_to
        self._tag = method.delivery_tag
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        self.do_work(msg.action)
        self._last_action = time_ns() // 1000 // 1000  # time in milliseconds

    def _to_db(self):
        with self.db_lock():
            with Session(self.engine) as session:
                for idx, conjecture in enumerate(self._conjectures):
                    if self._definitions[idx] is None:
                        self._definitions[idx] = []
                    session.add(
                        SQLConjecture(informal_problem=conjecture.informal_problem,
                                      parent_problem=conjecture.parent_problem,
                                      formal_problem=conjecture.formal_problem,
                                      definitions=[file.content for file in self._definitions[idx]],
                                      attempts=self._attempts[idx]))
                session.commit()

    def _handle_done(self):
        curr = time_ns() // 1000 // 1000
        diff = curr - self._last_action
        assert self._action is not None
        assert len(self._conjectures) == len(self._attempts)
        assert len(self._definitions) == len(self._conjectures)
        self._to_db()
        response = ResponseConjecture(action=self._action, conjectures=self._conjectures, attempts=self._attempts,
                                      ms_between=diff)
        msg = WorkerToMaster(message=WorkerResponse(response=response))
        self.channel.basic_publish(exchange="", routing_key=self._reply_to, body=msg.model_dump_json())
        logger.info("Completed message!")
        logger.debug("Reply to: %s", self._reply_to)
        self.channel.basic_ack(delivery_tag=self._tag)

    def do_work(self, action: ActionConjecture):
        assert action.conjecture.formal_problem is not None
        if action.conjecture.informal_proof is None:
            assert action.conjecture.parent_problem is not None, "One of the root theorems has no proof"
            informal_proof = sketch_proof(action.conjecture.formal_problem, action.conjecture.informal_problem,
                                          action.conjecture.parent_problem)
        else:
            informal_proof = action.conjecture.informal_proof
        conjectures = generate_conjectures(action.conjecture.informal_problem, informal_proof, model=self._model)
        # Deduplicate
        conjectures = list(dict.fromkeys(conjectures))
        self._attempts = [0] * len(conjectures)
        self._conjectures = [Conjecture(informal_problem=conjecture, parent_problem=action.conjecture.informal_problem) for conjecture in conjectures]
        self._action = action
        self._done = [False] * len(conjectures)
        self._definitions = [[] for _ in range(len(conjectures))]
        if not conjectures:
            logger.error("Did not generate any conjectures!")
            self._handle_done()
            return
        # Start the sub-conjectures
        for idx, conjecture in enumerate(self._conjectures):
            self._handle_conjecture(conjecture, idx)

    def _send_to_formalizer(self, msg: MasterToWorker):
        self.queues.formalization.send(self.channel, msg, self.formalizer_reply_to)

    def _handle_conjecture(self, conjecture: Conjecture, idx: int):
        assert self._action is not None
        self._send_to_formalizer(
            MasterToWorker(message=WorkerAction(
                action=ActionFormalization(conjecture=conjecture, run_config=self._action.run_config, search_idx=idx,
                                           definition_retrieval=self._action.definition_retrieval,
                                           num_formalizations=self._num_formalizations)
            ))
        )

    def _formalize_callback(self, ch, method, properties, body):
        to_master = WorkerToMaster.model_validate_json(body)
        msg = to_master.message
        assert isinstance(msg, WorkerResponse)
        logger.info("Received formalization callback!")
        response = msg.response
        if not isinstance(response, ResponseFormalization):
            raise RuntimeError("Response received was not a ResponseFormalization, might be the wrong queue!")
        logger.debug("Formal theorem: %s", response.formal_theorem)
        self._done[response.action.search_idx] = True
        self._conjectures[response.action.search_idx].formal_problem = response.formal_theorem
        self._definitions[response.action.search_idx] = response.definitions
        self._attempts[response.action.search_idx] = response.attempts
        if all(self._done):
            self._handle_done()
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    from os import getenv
    model = getenv("CONJECTURE_MODEL")
    if model is None:
        raise ValueError("Conjecture model not set!")
    db_path = getenv("SQLITE_PATH")
    if db_path is None:
        raise ValueError("Database path not set!")
    worker = ConjectureWorker(model, db_path)
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Conjecturer")
        raise e
    send_notification(False, job_name="Conjecturer")
