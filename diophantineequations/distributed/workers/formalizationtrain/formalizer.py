from diophantineequations.distributed.messages import MasterToWorker, RequestModelUpdate, WorkerToMaster, RunConfig, \
    WorkerRequest
from diophantineequations.distributed.workers.abstract import Worker
from diophantineequations.distributed.workers.formalizationtrain.train import FormalizationSample, \
    ActionFormalizationSample, Prompt
from diophantineequations.notify import send_notification
import logging
import subprocess
from sqlmodel import SQLModel, create_engine, Session, select, func
from functools import partial
import threading
from pathlib import Path
from typing import Optional
import traceback
from pika.exceptions import ConnectionClosed
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainWorker(Worker):
    def __init__(self, return_exchange: str, db_path: str, store_dir: str, deepspeed_bin: str,
                 deepspeed_config_path: str, num_new_samples: int = 512): # num_new_samples needs to be in sync with deepspeedconfig, i.e. num_gpus * device_batch_size * grad_acc_steps
        assert Path(deepspeed_bin).exists()
        logger.info("Initializing train worker")
        super().__init__(prefetch_count=num_new_samples)
        self.queues.formalization_samples.declare(self.channel)
        self.consumer_tag = self.queues.formalization_samples.consume(self.channel, self.callback)
        self.previous_model_type = None

        self.channel.exchange_declare(exchange=return_exchange, exchange_type='fanout', auto_delete=False)
        self.return_key = return_exchange
        self.db_path = db_path
        self.engine = create_engine(db_path)
        SQLModel.metadata.create_all(self.engine)
        self.samples = []
        self.delivery_tag = -1
        self.num_samples = num_new_samples
        self.sample_count = -1
        self.store_dir = store_dir
        self.deepspeed_path = deepspeed_bin
        self.deepspeed_config = deepspeed_config_path
        self._run_config: Optional[RunConfig] = None
        self._threaded = partial(self.threaded_work, work=self.do_work, final_hook=self._final_hook,
                                 send_reply_factory=self.send_reply)
        self._version = self._get_version()

    def _get_version(self) -> int:
        logger.info("Getting current version from db!")
        with Session(self.engine) as session:
            stmt = select(func.max(Prompt.version))
            version = session.exec(stmt).first()
        if version is None:
            version = 0
        logger.info("Current version: %s", version)
        return version

    def _get_sample_count(self) -> int:
        logger.info("Loading initial sample count!")
        with Session(self.engine) as session:
            stmt = (select(func.count(Prompt.id)).where(
                Prompt.run_config == self._run_config.model_dump(mode="json")).where(Prompt.version == self._version))
            sample_count = session.exec(stmt).first()
        if sample_count is None:
            sample_count = 0
        logger.info("Sample count: %s", sample_count)
        return sample_count

    def to_database(self, actions: list[ActionFormalizationSample], version: int):
        with Session(self.engine) as session:
            for action in actions:
                prompt, samples = FormalizationSample.from_action(action, version)
                session.add_all(samples)
            session.commit()
        self.sample_count += len(actions)

    def get_callback(self):
        return self._callback

    def _callback(self, ch, method, properties, body):
        logger.info("Received sample!")
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        # Once at the beginning, we can only init samples once we know our run config
        if self.sample_count < 0:
            assert self._run_config is None
            assert not self.samples
            self._run_config = msg.action.run_config
            self.sample_count = self._get_sample_count()
        assert self._run_config is not None
        assert self._run_config == msg.action.run_config, "Multiple different run configs in a trainer are not supported!"
        if not isinstance(msg.action, ActionFormalizationSample):
            raise RuntimeError("Action received was not an ActionFormalizationSample, might be the wrong queue!")
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        self.delivery_tag = max(self.delivery_tag, method.delivery_tag)
        self.samples.append(msg.action)
        if len(self.samples) < self.num_samples:
            logger.debug("At %s samples which is < %s", len(self.samples), self.num_samples)
            return None
        logger.info("Continuing onwards!")
        # To db runs in main thread, because we edit samples here. The rest is detached
        self._to_db()
        self.channel.basic_cancel(consumer_tag=self.consumer_tag)  # Stop receiving samples
        # Acknowledge before training to prevent writing samples twice
        self.channel.basic_ack(delivery_tag=self.delivery_tag, multiple=True)
        logger.debug("Acknowledged up to tag %s", self.delivery_tag)
        thread_fn = partial(self._threaded, ch, method, properties, action=msg.action)
        thread = threading.Thread(target=thread_fn, daemon=False)
        thread.start()
        return None

    def _start_consumer(self) -> None:
        """Subscribe to the queue (called at startup and after every batch)."""
        self.queues.formalization_samples.consume(self.channel, self.callback, consumer_tag=self.consumer_tag)
        logger.info("Consuming with tag %s", self.consumer_tag)

    def _to_db(self):
        logger.info("Writing %s samples to database %s", len(self.samples), self.db_path)
        self.to_database(self.samples, self._version)
        self.samples = []

    def _final_hook(self):
        def _resume():
            self._start_consumer()
            self.delivery_tag = -1
            logger.info("Resumed channel flow after training")

        self.connection.add_callback_threadsafe(_resume)

    def send_reply(self, ch, method, properties, msg: WorkerToMaster):
        def send_request():
            logger.info("Sending request %s", msg.model_dump_json())
            self.channel.basic_publish(exchange=self.return_key, routing_key="", body=msg.model_dump_json())
            ch.basic_publish(exchange="", routing_key=properties.reply_to, body=msg.model_dump_json())

        return send_request

    def do_work(self, action: ActionFormalizationSample) -> WorkerRequest:
        logger.debug("Starting training for model type %s stored at %s", action.model_type, action.model_path)
        result_path = Path(self.store_dir) / f"{self.sample_count}/"
        # Run accelerate launch in a subprocess
        train_script = Path(__file__).parent / "train.py"
        cmd = [self.deepspeed_path, str(train_script),
               str(self.db_path), str(action.model_path), str(result_path), "--run-config",
               action.run_config.model_dump_json(), "--version", str(self._version),
               "--deepspeed_config", self.deepspeed_config]
        logger.info("Executing: %s", " ".join(cmd))
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                logger.info("[trainer] %s", line.rstrip())
        if proc.returncode != 0:
            shutil.rmtree(str(result_path), ignore_errors=True)
            raise RuntimeError(
                f"accelerate launch failed (exit={proc.returncode}). \n, See the log above for details.")
        # Sanity check, there should be files
        if not any(result_path.glob("*")):
            raise RuntimeError(f"Training finished but no files appeared in {result_path}")

        logger.info("Training finished successfully, stored at %s", result_path)
        self._version += 1
        return WorkerRequest(request=RequestModelUpdate(model_path=str(result_path), model_type=action.model_type,
                                                        sample_count=self.sample_count))

    def start(self):
        logger.info("Starting worker %s", self.__class__.__name__)
        while True:
            try:
                self.channel.start_consuming()
            except KeyboardInterrupt as e:
                self.channel.stop_consuming()
                raise e
            except ConnectionClosed as e:
                logger.exception("AMQP connection closed – exiting")
                raise e
            else:
                self.connection.process_data_events(5)
                continue


def main():
    from os import getenv
    db_url = getenv("POSTGRES_URL")
    if db_url is None:
        raise ValueError("Database path not set!")
    return_queue = getenv("RABBIT_FORMALIZATION_TRAINER")
    if return_queue is None:
        raise ValueError("Return queue not set!")
    store_dir = getenv("MODEL_DIR")
    if store_dir is None:
        raise ValueError("Store dir not set!")
    if not Path(store_dir).exists():
        raise ValueError("Store dir does not exist!")
    deepspeed_config_path = getenv("DEEPSPEED_CONFIG")
    if deepspeed_config_path is None:
        raise ValueError("Deepspeed config not set!")
    if not Path(deepspeed_config_path).exists():
        raise ValueError("Deepspeed config file does not exist!")
    deepspeed_bin_path = getenv("DEEPSPEED_BIN")
    if deepspeed_bin_path is None:
        raise ValueError("Deepspeed bin not set!")
    if not Path(deepspeed_bin_path).exists():
        raise ValueError("Deepspeed bin does not exist!")
    worker = TrainWorker(return_queue, db_url, store_dir, deepspeed_bin_path, deepspeed_config_path,
                         num_new_samples=1000)
    worker.start()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "FormalizationTrainer")
        raise e
    send_notification(False, job_name="FormalizationTrainer")
