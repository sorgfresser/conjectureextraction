import os
from dataclasses import dataclass
from pika.channel import Channel
import pika
from pydantic import BaseModel
from typing import Optional, Callable
from typing_extensions import Self


@dataclass
class Queue:
    name: str
    auto_delete: bool
    durable: bool
    arguments: dict = None
    delivery_mode: int = 1

    def __post_init__(self):
        self.arguments = {} if self.arguments is None else self.arguments

    def declare(self, channel: Channel):
        channel.queue_declare(self.name, durable=self.durable, auto_delete=self.auto_delete, arguments=self.arguments)

    def send(self, channel: Channel, message: BaseModel, reply_to: Optional[str] = None):
        properties = pika.BasicProperties(reply_to=reply_to, delivery_mode=self.delivery_mode)
        channel.basic_publish("", routing_key=self.name, body=message.model_dump_json(), properties=properties)

    def consume(self, channel: Channel, callback: Callable, consumer_tag: Optional[str] = None) -> str:
        return channel.basic_consume(queue=self.name, on_message_callback=callback, auto_ack=False, consumer_tag=consumer_tag)


class Queues:
    def __init__(self, search: str, environment: str, tactics: str, tactics_samples: str, conjecture: str,
                 formalization: str, formalization_samples: str, embed: str, retrieve: str, search_result: str):
        self.search = Queue(search, False, True, {'x-consumer-timeout': 36_000_000},
                            delivery_mode=2)  # 36000 seconds, i.e. 10 hours
        self.environment = Queue(environment, False, False)
        self.tactics = Queue(tactics, False, False)
        self.tactics_samples = Queue(tactics_samples, False, True, {'x-consumer-timeout': 36_000_000}, delivery_mode=2)
        self.conjecture = Queue(conjecture, False, True, {'x-consumer-timeout': 604_800_000}, delivery_mode=2)
        self.formalization = Queue(formalization, False, False)
        self.formalization_samples = Queue(formalization_samples, False, True, {'x-consumer-timeout': 36_000_000},
                                           delivery_mode=2)
        self.embed = Queue(embed, False, True, {'x-consumer-timeout': 36_000_000}, delivery_mode=2)
        self.retrieve = Queue(retrieve, False, True, delivery_mode=2)
        self.search_result = Queue(search_result, False, True, {'x-consumer-timeout': 36_000_000}, delivery_mode=2)

    @classmethod
    def from_env_vars(cls) -> Self:
        search = os.environ["RABBIT_SEARCH"]
        environment = os.environ["RABBIT_ENVIRONMENT"]
        tactics = os.environ["RABBIT_TACTICS"]
        tactic_samples = os.environ["RABBIT_TACTIC_SAMPLES"]
        conjecture = os.environ["RABBIT_CONJECTURE"]
        formalization = os.environ["RABBIT_FORMALIZATION"]
        formalization_samples = os.environ["RABBIT_FORMALIZATION_SAMPLES"]
        embed = os.environ["RABBIT_EMBED"]
        retrieve = os.environ["RABBIT_RETRIEVE"]
        search_result = os.environ["RABBIT_SEARCH_RESULT"]
        return cls(search, environment, tactics, tactic_samples, conjecture, formalization, formalization_samples, embed,
            retrieve, search_result)
