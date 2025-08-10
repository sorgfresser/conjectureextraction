from diophantineequations.distributed.workers.abstract import Worker
from diophantineequations.distributed.messages import ActionFormalization, Conjecture, MasterToWorker, WorkerToMaster, \
    WorkerResponse, ResponseFormalization, ResponseInitialEnvironment, WorkerAction, ActionInitialEnvironment, \
    WorkerRequest, RequestModelUpdate
from abc import ABC, abstractmethod
from diophantineequations.utils import is_valid
from diophantineequations.definition_retrieval import DefinitionVectorStore, informalize, LeanFile
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from logging import getLogger
from functools import partial
import threading
from time import time_ns
from typing import Optional
from pathlib import Path
from os import getenv
from sqlmodel import SQLModel, Field, Session, select, func, create_engine
from sqlalchemy.exc import IntegrityError

logger = getLogger(__name__)


class FormalizerVersion(SQLModel, table=True):
    id: int = Field(primary_key=True)
    model_path: str


class AbstractFormalizationWorker(Worker, ABC):
    def __init__(self, trainer_exchange: str, initial_model_path: str, db_path: str, definition_path: Path,
                 max_attempts: int = 10):
        super().__init__(prefetch_count=1)
        self.queues.formalization.declare(self.channel)
        self.consumer_tag = self.queues.formalization.consume(self.channel, self.callback)
        self._threaded = partial(self.threaded_work, work=self.do_work, send_reply_factory=self.send_reply)
        # Internal
        self.model_path = initial_model_path
        # Db path
        self.db_path = db_path
        self.engine = create_engine(db_path)
        SQLModel.metadata.create_all(self.engine)
        version = self._latest_model_version()
        if version is not None:
            logger.info("Found latest version %s", version)
            self.model_path = version.model_path
        self._attempts = 0
        self._max_attempts = max_attempts
        self._action: Optional[ActionFormalization] = None
        self._formalizations: list[Optional[str]] = [None for _ in range(max_attempts)]
        self._working: list[bool] = [False for _ in range(max_attempts)]
        self._tag = None
        self._reply_to = None
        self._definition_path = definition_path
        self.definition_vector_store = DefinitionVectorStore.from_file(DefaultEmbeddingFunction(), informalize)
        if self.definition_vector_store is None:
            self.definition_vector_store = DefinitionVectorStore.from_directory(definition_path,
                                                                                DefaultEmbeddingFunction(), informalize)
        self._definitions = []
        # Env queue
        self.environment_reply_to = self.get_exclusive(self._env_callback)
        self.queues.environment.declare(self.channel)
        # Trainer samples
        self.queues.formalization_samples.declare(self.channel)
        # Trainer response
        result = self.get_exclusive(self._trained_callback)
        self.trainer_exchange = trainer_exchange
        self.channel.exchange_declare(trainer_exchange, exchange_type="fanout")
        self.channel.queue_bind(queue=result, exchange=trainer_exchange)
        # Stats
        self._last_action = time_ns() // 1000 // 1000
        self._received_time = time_ns() // 1000 // 1000

    def get_callback(self):
        return self._callback

    def _callback(self, ch, method, properties, body) -> None:
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        if not isinstance(msg.action, ActionFormalization):
            raise RuntimeError("Action received was not an ActionFormalization, might be the wrong queue!")
        logger.info("Received new ActionFormalization!")
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        self._tag = method.delivery_tag
        self._reply_to = properties.reply_to
        thread_fn = partial(self._threaded, ch, method, properties, action=msg.action)
        thread = threading.Thread(target=thread_fn)
        thread.start()

    def _latest_model_version(self) -> Optional[FormalizerVersion]:
        # Model with the highest sample count
        with Session(self.engine) as session:
            stmt = select(FormalizerVersion).where(select(func.max(FormalizerVersion.id)).scalar_subquery() == FormalizerVersion.id)
            version = session.exec(stmt).first()
        return version

    def _trained_callback(self, ch, method, properties, body):
        msg = WorkerToMaster.model_validate_json(body)
        assert isinstance(msg.message, WorkerRequest)
        assert isinstance(msg.message.request, RequestModelUpdate)
        request = msg.message.request
        self.model_path = request.model_path
        logger.info("New model received at %s", self.model_path)
        # Only add once, otherwise ignore
        with Session(self.engine) as session:
            logger.info("Setting latest model in database for id %s!", request.sample_count)
            try:
                session.add(FormalizerVersion(
                    model_path=self.model_path,
                    id=request.sample_count
                ))
                session.commit()  # one worker commits, rest fails here
                logger.info("Stored model id %s in DB", request.sample_count)
            except IntegrityError:
                session.rollback()
                logger.info("Model version already present, skipping insert.")

        ch.basic_ack(method.delivery_tag)
        self._last_action = time_ns() // 1000 // 1000

    def send_reply(self, ch, method, properties, m: None):
        def noop():
            pass

        return noop

    def send_response(self, formalization: Optional[str], attempt: int, conjecture: Conjecture):
        msg = WorkerToMaster(message=WorkerResponse(
            response=ResponseFormalization(
                action=self._action,
                attempts=attempt,
                formal_theorem=formalization,
                ms_between=self._received_time - self._last_action,
                definitions=self._definitions,
            )
        ))
        logger.debug("Sending response %s", msg.model_dump_json())
        self.channel.basic_publish(exchange="", routing_key=self._reply_to, body=msg.model_dump_json())
        self.channel.basic_ack(delivery_tag=self._tag)
        self._last_action = time_ns() // 1000 // 1000  # time in milliseconds

    def do_work(self, action: ActionFormalization):
        self._received_time = time_ns() // 1000 // 1000
        assert isinstance(action, ActionFormalization)
        logger.info("Starting formalization!")
        logger.debug("Conjecture: %s", action.conjecture)
        self._attempts = 0
        self._action = action
        self._working: list[bool] = [False for _ in range(self._max_attempts)]
        lean_files = []
        if self._action.definition_retrieval:
            lean_files = self.definition_vector_store.get_definitions(action.conjecture.informal_problem, 10)
            logger.debug("Retrieved definitions %s", lean_files)
        logger.info("Generating formalization candidates!")
        self._formalizations = self.get_formalizations(action.conjecture, lean_files)
        logger.debug("Formalized %s", self._formalizations)
        self._definitions = lean_files
        if not self.enqueue_next_environment():
            self.handle_done(action.conjecture)

    def _env_callback(self, ch, method, properties, body):
        logger.info("Received env callback!")
        msg = WorkerToMaster.model_validate_json(body, by_name=True)
        assert isinstance(msg.message, WorkerResponse)
        response = msg.message.response
        assert isinstance(response, ResponseInitialEnvironment)
        assert response.action.run_config == self._action.run_config
        # Formalization successful
        if response.proof_state is not None:
            self._working[self._attempts] = True
        self._attempts += 1
        if self._attempts >= self._max_attempts:
            self.handle_done(self._action.conjecture)
        elif not self.enqueue_next_environment():
            self.handle_done(self._action.conjecture)
        logger.info("Completed message!")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def enqueue_next_environment(self) -> bool:
        assert self._attempts < self._max_attempts
        assert len(self._formalizations) == self._max_attempts
        formalized = self._formalizations[self._attempts:]
        for formalproblem in formalized:
            # If the string parsing failed, just increment attempts
            if formalproblem is None:
                self._attempts += 1
                continue
            # If for some reason it is unparseable
            if not is_valid(formalproblem):
                self._attempts += 1
                continue
            msg = MasterToWorker(
                message=WorkerAction(
                    action=ActionInitialEnvironment(theorem=formalproblem, search_idx=self._action.search_idx,
                                                    run_config=self._action.run_config, past_tactics=[], context=None)))
            self._send_to_environment(msg)
            return True
        return False

    def _send_to_trainer(self, msg: MasterToWorker):
        self.queues.formalization_samples.send(self.channel, msg)

    @abstractmethod
    def get_formalizations(self, conjecture: Conjecture, definitions: list[LeanFile]) -> list[Optional[str]]:
        pass

    def handle_done(self, conjecture: Conjecture) -> None:
        def send():
            logger.debug("Sending response!")
            self.send_response(formalization, attempt, conjecture)
        try:
            idx = self._working.index(True)
        except ValueError:
            idx = None
        formalization = None if idx is None else self._formalizations[idx]
        attempt = self._max_attempts if idx is None else idx + 1.0
        logger.info("Finished formalization!")
        logger.debug("Formalized %s", formalization)
        self.connection.add_callback_threadsafe(send)

    def _send_to_environment(self, msg: MasterToWorker):
        def send():
            logger.debug("Sending message %s to environment", msg.model_dump_json())
            self.queues.environment.send(self.channel, msg, self.environment_reply_to)

        logger.info("Sending to environment!")
        self.connection.add_callback_threadsafe(send)

    @classmethod
    def from_env_vars(cls):
        definition_path = getenv("DEFINITION_PATH")
        if definition_path is None:
            raise ValueError("Definition path not set!")
        definition_path = Path(definition_path)
        if not definition_path.exists():
            raise ValueError("Definition path not found!")
        trainer_exchange = getenv("RABBIT_FORMALIZATION_TRAINER")
        if trainer_exchange is None:
            raise ValueError("Trainer exchange not set!")
        db_url = getenv("POSTGRES_URL")
        if db_url is None:
            raise ValueError("Database path not set!")
        initial_model_path = getenv("MODEL_PATH")
        if initial_model_path is None:
            raise ValueError("Initial model path not provided!")

        return cls(trainer_exchange, initial_model_path, db_url, definition_path)
