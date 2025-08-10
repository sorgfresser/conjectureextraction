import pika
from abc import ABC, abstractmethod
import logging
from os import getenv
from typing import Callable, overload
from pika.amqp_object import Properties, Method
from pika.spec import Channel
from pika.exceptions import ChannelClosed
from diophantineequations.distributed.messages import AbstractAction, WorkerResponse, WorkerToMaster, WorkerRequest
from diophantineequations.distributed.queues import Queues
from diophantineequations.notify import send_notification
import traceback

logger = logging.getLogger(__name__)


class RecoverableError(Exception):
    pass


class GenericConnector(ABC):
    def __init__(self, non_blocking: bool = False):
        username = getenv("RABBIT_USER", "guest")
        password = getenv("RABBIT_PASSWORD", "guest")
        host = getenv("RABBIT_HOST", "localhost")
        credentials = pika.PlainCredentials(username, password)
        if non_blocking:
            self.connection = pika.SelectConnection(pika.ConnectionParameters(host=host, credentials=credentials))
        else:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
        self.channel = self.connection.channel()


class Worker(GenericConnector):
    def __init__(self, non_blocking: bool = False, prefetch_count: int = 3):
        logger.info("Initializing worker %s", self.__class__.__name__)
        super().__init__(non_blocking)
        self.queues = Queues.from_env_vars()
        self.channel.basic_qos(prefetch_count=prefetch_count)
        self.callback = self.get_callback()
        self.is_blocking = not non_blocking

    @abstractmethod
    def get_callback(self):
        pass

    def start(self):
        logger.info("Starting worker %s", self.__class__.__name__)
        if self.is_blocking:
            self.channel.start_consuming()

    @overload
    def threaded_work(self, ch, method, properties, action: AbstractAction,
                      work: Callable[[AbstractAction], None],
                      send_reply_factory: Callable[[Channel, Method, Properties, None], Callable[
                          [], None]] | None = None,
                      final_hook: Callable[[], None] | None = None) -> None:
        ...

    def threaded_work(self, ch, method, properties, action: AbstractAction,
                      work: Callable[[AbstractAction], WorkerResponse | WorkerRequest],
                      send_reply_factory: Callable[[Channel, Method, Properties, WorkerToMaster], Callable[
                          [], None]] | None = None,
                      final_hook: Callable[[], None] | None = None) -> None:
        try:
            result = work(action)
            msg = WorkerToMaster(message=result) if result is not None else None
            if send_reply_factory is not None:
                logger.debug("Sending reply %s", msg.model_dump_json() if msg is not None else None)
                send_reply = send_reply_factory(ch, method, properties, msg)
                self.connection.add_callback_threadsafe(send_reply)
        except pika.exceptions.ChannelClosed:
            logger.exception("Channel closed in %s!", self.__class__.__name__)
            send_notification(True, traceback.format_exc(), self.__class__.__name__)
            exit(1)
        except Exception as e:
            def send_nack():
                logger.debug("Sending nack for action %s", action.model_dump_json())
                ch.basic_nack(delivery_tag=method.delivery_tag)

            if isinstance(e, RecoverableError):
                logger.exception("Recoverable exception in %s!", self.__class__.__name__)
            else:
                logger.exception("Exception in %s!", self.__class__.__name__)
                send_notification(True, traceback.format_exc(), self.__class__.__name__)
            self.connection.add_callback_threadsafe(send_nack)
        finally:
            if final_hook is not None:
                final_hook()

    def get_exclusive(self, callback: Callable) -> str:
        """Get a new exclusive queue and consume it with callback

        :param callback:
        :return: The queue name of the newly created queue.
        """
        result = self.channel.queue_declare(queue="", exclusive=True)
        reply_to = result.method.queue
        self.channel.basic_consume(queue=reply_to, on_message_callback=callback,
                                   auto_ack=False)
        return reply_to
