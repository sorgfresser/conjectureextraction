from diophantineequations.distributed.workers.abstract import Worker
from diophantineequations.distributed.messages import MasterToWorker, WorkerResponse, \
    WorkerToMaster, ActionRetrieve, WorkerAction, ActionInitialEnvironment, ResponseInitialEnvironment, ResponseRetrieve
from diophantineequations.lemma_embeddings import LemmaVectorStore, ReProverEmbeddingFn
from diophantineequations.notify import send_notification
from lean_interact.interface import Sorry
import logging
from time import time_ns
from functools import partial
import threading
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalWorker(Worker):
    def __init__(self, collection_name: Optional[str]):
        # Retrieve
        super().__init__(prefetch_count=1)
        self.queues.retrieve.declare(self.channel)
        self.consumer_tag = self.queues.retrieve.consume(self.channel, self.callback)
        if collection_name:
            logger.info("Using collection %s", collection_name)
            self._vector_store = LemmaVectorStore.from_embedding_fn(ReProverEmbeddingFn(),
                                                                    collection_name=collection_name)
        else:
            self._vector_store = LemmaVectorStore.from_embedding_fn(ReProverEmbeddingFn())
        self._threaded_retrieve = partial(self.threaded_work, work=self.retrieve, send_reply_factory=self.send_reply)
        # Internals
        self._last_action = time_ns() // 1000 // 1000
        self._reply_to = None
        self._tag = None
        self._action: Optional[ActionRetrieve] = None
        # Env
        self.environment_reply_to = self.get_exclusive(self._env_callback)
        self.queues.environment.declare(self.channel)
        self._awaiting_env = 0

    def get_callback(self):
        return self._retrieve_callback

    def _retrieve_callback(self, ch, method, properties, body):
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        if not isinstance(msg.action, ActionRetrieve):
            raise RuntimeError("Action received was not an ActionRetrieve, might be the wrong queue!")
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        self._reply_to = properties.reply_to
        self._tag = method.delivery_tag
        self._action = msg.action
        logger.info("Sending to environment!")
        self._send_to_environment(MasterToWorker(
            message=WorkerAction(action=ActionInitialEnvironment(
                theorem=msg.action.theorem,
                past_tactics=msg.action.past_tactics,
                search_idx=msg.action.search_idx,
                run_config=msg.action.run_config,
                context=None))))

    def _env_callback(self, ch, method, properties, body):
        logger.info("Received environment callback!")
        diff = time_ns() // 1000 // 1000
        diff = diff - self._last_action
        from_master = WorkerToMaster.model_validate_json(body)
        msg = from_master.message
        if not isinstance(msg, WorkerResponse) or not isinstance(msg.response, ResponseInitialEnvironment):
            raise RuntimeError("Response received was not an ResponseEnvironment, might be the wrong queue!")
        response = msg.response
        if response.action.run_config.dry_run:
            logger.info("Performing dry run!")
        if response.proof_state is None:
            msg = WorkerToMaster(
                message=WorkerResponse(response=ResponseRetrieve(action=self._action, files=[], ms_between=diff))
            )
            logger.info("No proof state, returning...")
            self.send_reply(ch, method, properties, msg)()
            return
        if isinstance(response.proof_state, Sorry):
            goal = response.proof_state.goal
        else:
            goal = "\n".join(response.proof_state.goals)
        action = self._action.model_copy(deep=True)
        action.goal = goal
        thread_fn = partial(self._threaded_retrieve, ch, method, properties, action=action)
        thread = threading.Thread(target=thread_fn)
        thread.start()

    def _send_to_environment(self, msg: MasterToWorker):
        self.queues.environment.send(self.channel, msg, self.environment_reply_to)

    def retrieve(self, action: ActionRetrieve) -> WorkerResponse:
        diff = time_ns() // 1000 // 1000
        diff = diff - self._last_action
        logger.info("Getting premises...")
        files = self._vector_store.get_premises(action.goal, action.k)
        logger.info("Retrieved %s premises, returning...", action.k)
        logger.debug("Premises: %s\nGoal: %s", files, action.goal)
        return WorkerResponse(response=ResponseRetrieve(action=action, files=files, ms_between=diff))

    def send_reply(self, ch, method, properties, msg: WorkerToMaster):
        def send_response():
            logger.debug("Sending response %s", msg.model_dump_json())
            ch.basic_publish(exchange="", routing_key=self._reply_to, body=msg.model_dump_json())
            ch.basic_ack(delivery_tag=self._tag)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        return send_response


def main():
    from os import getenv
    collection_name = getenv("LEMMA_COLLECTION")
    worker = RetrievalWorker(collection_name)
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Retrieval")
        raise e
    send_notification(False, job_name="Retrieval")
