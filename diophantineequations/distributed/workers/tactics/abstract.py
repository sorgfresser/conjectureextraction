from diophantineequations.distributed.messages import MasterToWorker, ActionTactics, ResponseTactics, WorkerResponse, \
    WorkerToMaster
from diophantineequations.distributed.workers.abstract import Worker
import logging
import threading
from time import time_ns
from abc import ABC, abstractmethod
from os import getenv
from functools import partial

logger = logging.getLogger(__name__)


class AbstractTacticWorker(Worker, ABC):
    def __init__(self):
        super().__init__(prefetch_count=1)
        self.queues.tactics.declare(self.channel)
        self.consumer_tag = self.queues.tactics.consume(self.channel, self.callback)
        self.previous_model_path = None
        self.previous_model_type = None
        self._last_action = time_ns() // 1000 // 1000  # time in milliseconds
        self._threaded = partial(self.threaded_work, work=self._work, send_reply_factory=self.send_reply)

    def _work(self, action: ActionTactics) -> WorkerResponse:
        diff = time_ns() // 1000 // 1000
        diff = diff - self._last_action
        self._get_model(action)
        response = self.do_work(action, diff)
        logger.debug("Tactics: %s", response.strings)
        logger.debug("Logprobs: %s", response.logprobs)
        return WorkerResponse(response=response)

    def get_callback(self):
        return self._callback

    def _callback(self, ch, method, properties, body) -> None:
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        if not isinstance(msg.action, ActionTactics):
            raise RuntimeError("Action received was not an ActionTactics, might be the wrong queue!")
        logger.info("Received new ActionTactics!")
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        thread_fn = partial(self._threaded, ch, method, properties, action=msg.action)
        thread = threading.Thread(target=thread_fn)
        thread.start()

    def send_reply(self, ch, method, properties, msg: WorkerToMaster):
        def send_response():
            logger.debug("Sending response %s", msg.model_dump_json())
            ch.basic_publish(exchange="", routing_key=properties.reply_to, body=msg.model_dump_json())
            ch.basic_ack(delivery_tag=method.delivery_tag)

        return send_response

    def _get_model(self, action: ActionTactics) -> None:
        if action.model_type == self.previous_model_type and action.model_path == self.previous_model_path:
            return
        logger.info("Models differ...")
        self.reload_model(action)
        self.previous_model_type = action.model_type
        self.previous_model_path = action.model_path

    @abstractmethod
    def reload_model(self, action: ActionTactics) -> None:
        pass

    @abstractmethod
    def do_work(self, action: ActionTactics, diff: int) -> ResponseTactics:
        pass

