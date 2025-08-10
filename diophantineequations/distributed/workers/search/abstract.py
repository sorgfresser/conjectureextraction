from sqlmodel import SQLModel, Field, Enum, PrimaryKeyConstraint, create_engine, Session, select, func
from diophantineequations.distributed.messages import ModelType
from diophantineequations.distributed.workers.abstract import Worker
from logging import getLogger
from time import time_ns
from typing import Optional
from abc import ABC, abstractmethod
from diophantineequations.distributed.messages import AbstractActionSearch, WorkerToMaster, WorkerRequest, \
    RequestModelUpdate, MasterToWorker, WorkerResponse, ResponseTactics, WorkerAction, ActionEnvironment, \
    ActionEnvironmentAddHypotheses, ResponseEnvironmentAddHypotheses, ActionEnvironmentWholeProof
import pika
from os import getenv
from sqlalchemy.exc import IntegrityError

logger = getLogger(__name__)




class ModelVersion(SQLModel, table=True):
    id: int = Field(primary_key=True)
    model_type: ModelType = Field(sa_type=Enum(ModelType), primary_key=True)
    model_path: str

    __table_args__ = (PrimaryKeyConstraint("id", "model_type"),)


class AbstractSearchWorker(Worker, ABC):
    def __init__(self, trainer_exchange: str, db_path: str, initial_model_path: str, initial_model_type: ModelType):
        super().__init__(prefetch_count=1)
        self.queues.search.declare(self.channel)
        self.consumer_tag = self.queues.search.consume(self.channel, self.callback)
        self.model_path = initial_model_path
        self.model_type = initial_model_type
        # Db path
        self.db_path = db_path
        self.engine = create_engine(db_path)
        # Race condition
        try:
            SQLModel.metadata.create_all(self.engine)
        except IntegrityError:
            pass
        version = self._latest_model_version()
        if version is not None:
            logger.info("Found latest version %s", version)
            self.model_type = version.model_type
            self.model_path = version.model_path

        # Internals
        self._goal = None
        self._tag = None
        self._action: Optional[AbstractActionSearch] = None
        self._reply_to = None
        self._current_idx = 0
        self._theorem_statement = None
        self._trial = 0
        self._last_action = time_ns() // 1000 // 1000  # time in milliseconds
        self._search_start_time = -1
        # Trainer samples
        self.queues.tactics_samples.declare(self.channel)
        # Trainer response
        result = self.get_exclusive(self._trained_callback)
        self.trainer_exchange = trainer_exchange
        self.channel.exchange_declare(trainer_exchange, exchange_type="fanout")
        self.channel.queue_bind(queue=result, exchange=trainer_exchange)
        # Env
        self.environment_reply_to = self.get_exclusive(self._env_callback)
        self.queues.environment.declare(self.channel)
        self._awaiting_env = 0
        # Tactics
        self.tactics_reply_to = self.get_exclusive(self._tactics_callback)
        self.queues.tactics.declare(self.channel)
        self._awaiting_tactics = 0

    def _latest_model_version(self) -> Optional[ModelVersion]:
        # Model with the highest sample count
        with Session(self.engine) as session:
            stmt = select(ModelVersion).where(select(func.max(ModelVersion.id)).scalar_subquery() == ModelVersion.id)
            version = session.exec(stmt).first()
        return version

    def _trained_callback(self, ch, method, properties, body):
        msg = WorkerToMaster.model_validate_json(body)
        assert isinstance(msg.message, WorkerRequest)
        assert isinstance(msg.message.request, RequestModelUpdate)
        request = msg.message.request
        self.model_path = request.model_path
        self.model_type = request.model_type
        logger.info("New model received at %s with type %s", self.model_path, self.model_type)
        # Only add once, otherwise ignore
        with Session(self.engine) as session:
            logger.info("Setting latest model in database for id %s!", request.sample_count)
            try:
                session.add(ModelVersion(
                    model_path=self.model_path,
                    model_type=request.model_type,
                    id=request.sample_count
                ))
                session.commit()  # one worker commits, rest fails here
                logger.info("Stored model id %s in DB", request.sample_count)
            except IntegrityError:
                session.rollback()
                logger.info("Model version already present, skipping insert.")

        ch.basic_ack(method.delivery_tag)
        self._last_action = time_ns() // 1000 // 1000

    def _env_callback(self, ch, method, properties, body) -> None:
        logger.info("Received env callback!")
        msg = WorkerToMaster.model_validate_json(body, by_name=True)
        assert isinstance(msg.message, WorkerResponse)
        if isinstance(msg.message.response, ResponseEnvironmentAddHypotheses):
            self.handle_added_hypotheses(msg.message.response)
        else:
            self.env_callback(msg.message)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        self._last_action = time_ns() // 1000 // 1000

    def get_callback(self):
        return self._search_callback

    def _search_callback(self, ch, method, properties, body):
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        if not isinstance(msg.action, AbstractActionSearch):
            raise RuntimeError("Action received was not an AbstractActionSearch, might be the wrong queue!")
        logger.info("Starting search for search idx %s", msg.action.search_idx)
        logger.debug("Received premises %s", msg.action.premises)
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")

        # Reset search state
        action = msg.action
        self._tag = method.delivery_tag
        self._reply_to = properties.reply_to
        self._search_start_time = time_ns()
        self._goal = None
        self._current_idx = 0
        self._action = action
        self._awaiting_env = 0
        self._awaiting_tactics = 0

        # Get theorem statement with premises as hypotheses
        msg = MasterToWorker(message=WorkerAction(action=ActionEnvironmentAddHypotheses(
            search_idx=self._action.search_idx,
            run_config=self._action.run_config,
            theorem=self._action.theorem,
            context=None,
            decls=self._action.premises,
        )))
        self._send_to_environment(msg)

        self._last_action = time_ns() // 1000 // 1000

    def _tactics_callback(self, ch, method, properties, body):
        logger.info("Received tactics callback!")
        msg = WorkerToMaster.model_validate_json(body)
        assert isinstance(msg.message, WorkerResponse)
        response = msg.message
        self._awaiting_tactics -= 1
        assert self._awaiting_tactics >= 0
        self.tactics_callback(response)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        self._last_action = time_ns() // 1000 // 1000

    @abstractmethod
    def do_work(self) -> None:
        pass

    @abstractmethod
    def env_callback(self, response: WorkerResponse) -> None:
        pass

    def tactics_callback(self, response: WorkerResponse) -> None:
        response = response.response
        assert isinstance(response, ResponseTactics)
        if response.ms_between > 3000:
            logger.warning("Tactic generation waited more than 3 seconds, consider changing the worker distribution!")
        context = "\n".join(response.action.premises) if response.action.premises else None
        logger.debug("Theorem %s with context: %s", self._theorem_statement, context)
        logger.debug("Tactics: %s", response.strings)
        # Whole proof
        if self.model_type == ModelType.deepseek:
            action = ActionEnvironmentWholeProof(
                search_idx=self._action.search_idx,
                run_config=self._action.run_config,
                theorem=self._theorem_statement,
                context=None, # todo: this is ugly, reactivate in theory, but we don't need it rn
                proofs=response.strings,
            )
        else:
            action = ActionEnvironment(
                theorem=self._theorem_statement, past_tactics=response.action.past_tactics, goal=response.action.goal,
                current_tactics=response.strings, current_logprobs=response.logprobs, search_idx=self._action.search_idx,
                run_config=self._action.run_config, context=context)
        msg = MasterToWorker(message=WorkerAction(action=action))
        self._send_to_environment(msg)

    def _send_to_environment(self, msg: MasterToWorker):
        self.queues.environment.send(self.channel, msg, self.environment_reply_to)

    def _send_response_to_root(self, msg: WorkerToMaster):
        self.channel.basic_publish(exchange="", routing_key=self._reply_to, body=msg.model_dump_json(),
                                   properties=pika.BasicProperties(delivery_mode=2))
        self.channel.basic_ack(delivery_tag=self._tag)

    def _send_to_tactic(self, msg: MasterToWorker):
        self.queues.tactics.send(self.channel, msg, self.tactics_reply_to)

    def _send_to_trainer(self, msg: MasterToWorker):
        self.queues.tactics_samples.send(self.channel, msg)

    def handle_added_hypotheses(self, response: ResponseEnvironmentAddHypotheses) -> None:
        # Do the actual work
        if response.error:
            logger.warning("Could not add hypotheses, ending with proven=False!")
            self.handle_done(False, 0, response.error)
            return
        self._theorem_statement = response.theorem
        self._action.theorem = self._theorem_statement
        self.do_work()


    @abstractmethod
    def handle_done(self, proven: bool, diff: int, error: Optional[str]) -> None:
        pass

    @classmethod
    def from_env_vars(cls):
        trainer_exchange = getenv("RABBIT_TRAINER_EXCHANGE")
        if trainer_exchange is None:
            raise ValueError("Trainer exchange not set!")
        initial_model_path = getenv("MODEL_PATH")
        if initial_model_path is None:
            raise ValueError("Initial model path not provided!")
        initial_model_type = getenv("MODEL_TYPE")
        if initial_model_type is None:
            raise ValueError("Initial model type not provided!")
        if initial_model_type == "deepseek":
            initial_model_type = ModelType.deepseek
        elif initial_model_type == "reprover":
            initial_model_type = ModelType.reprover
        else:
            raise ValueError("Model type must be deepseek or reprover!")
        db_url = getenv("POSTGRES_URL")
        if db_url is None:
            raise ValueError("Database path not set!")

        return cls(trainer_exchange, db_url, initial_model_path, initial_model_type)
