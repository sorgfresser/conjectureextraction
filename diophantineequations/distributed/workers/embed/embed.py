from diophantineequations.distributed.workers.abstract import Worker
from diophantineequations.distributed.messages import MasterToWorker, ActionEmbed, ResponseEmbed, WorkerResponse, \
    WorkerToMaster
from diophantineequations.lemma_embeddings import LemmaVectorStore, ReProverEmbeddingFn
from diophantineequations.lemma_add import add_conjecture
from diophantineequations.notify import send_notification
import logging
from time import time_ns
from functools import partial
import threading
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbedWorker(Worker):
    def __init__(self, lean_root_path: Path, source_path: Path, base_file: Path, collection_name: Optional[str]):
        assert base_file.exists()
        # Embed
        super().__init__(prefetch_count=1)
        self.queues.embed.declare(self.channel)
        self.consumer_tag = self.queues.embed.consume(self.channel, self.callback)
        if collection_name:
            logger.info("Using collection %s", collection_name)
            self._vector_store = LemmaVectorStore.from_directory(source_path, ReProverEmbeddingFn(),
                                                                 collection_name=collection_name)
        else:
            self._vector_store = LemmaVectorStore.from_directory(source_path, ReProverEmbeddingFn())
        self._threaded_embed = partial(self.threaded_work, work=self.embed, send_reply_factory=self.send_reply)
        # Internals
        self._last_action = time_ns() // 1000 // 1000
        self._root_path = lean_root_path
        self._source_path = source_path
        self._base_file = base_file

    def get_callback(self):
        return self._embed_callback

    def _embed_callback(self, ch, method, properties, body):
        from_master = MasterToWorker.model_validate_json(body)
        msg = from_master.message
        if not isinstance(msg.action, ActionEmbed):
            raise RuntimeError("Action received was not an ActionEmbed, might be the wrong queue!")
        if msg.action.run_config.dry_run:
            logger.info("Performing dry run!")
        thread_fn = partial(self._threaded_embed, ch, method, properties, action=msg.action)
        thread = threading.Thread(target=thread_fn)
        thread.start()

    def embed(self, action: ActionEmbed) -> WorkerResponse:
        diff = time_ns() // 1000 // 1000
        diff = diff - self._last_action
        assert action.theorem
        logger.info("Embedding with imports %s", action.theorem.imports)
        if not action.theorem.imports:
            action.theorem.imports = ["import Mathlib", "import Aesop", "open BigOperators Real Nat Topology Rat"]
            logger.info("No imports present, using imports %s", action.theorem.imports)
        for line in action.theorem.formalized_conjecture.splitlines():
            # We disallow defs for now, since that messes with the hypothesis injection
            if line.startswith("def "):
                return WorkerResponse(response=ResponseEmbed(success=False, action=action, ms_between=diff))
        file_path = add_conjecture(self._root_path, self._source_path, self._base_file, action.theorem, wrap_namespace=True)
        if not file_path.exists():
            logger.warning("Could not add conjecture, skipping adding to vector store!")
            logger.info("Conjecture that could not be added: %s", action.theorem)
            send_notification(True, f"Embedding failed!\n{action.theorem}", "Embed")
            return WorkerResponse(response=ResponseEmbed(success=False, action=action, ms_between=diff))
        self._vector_store.add_single_file(file_path, attempts=action.conjecture.attempts)
        logger.info("Added conjecture to the vector store, returning...")
        self._last_action = time_ns() // 1000 // 1000
        return WorkerResponse(response=ResponseEmbed(success=True, action=action, ms_between=diff))

    @staticmethod
    def send_reply(ch, method, properties, msg: WorkerToMaster):
        def send_response():
            logger.debug("Sending response %s", msg.model_dump_json())
            ch.basic_publish(exchange="", routing_key=properties.reply_to, body=msg.model_dump_json())
            ch.basic_ack(delivery_tag=method.delivery_tag)

        return send_response


def main():
    from os import getenv
    lean_root_path = getenv("LEAN_ROOT")
    if lean_root_path is None:
        raise ValueError("Lean root path not set!")
    lean_root_path = Path(lean_root_path)
    if not lean_root_path.exists():
        raise ValueError("Lean root path not found!")
    source_path = getenv("LEAN_SOURCE")
    if source_path is None:
        raise ValueError("Lean source path not set!")
    source_path = Path(source_path)
    if not source_path.exists():
        raise ValueError("Source path does not exist!")
    base_file = getenv("LEAN_FILE")
    if base_file is None:
        raise ValueError("Base file not set!")
    base_file = Path(base_file)
    if not base_file.exists():
        raise ValueError("Base file does not exist!")
    collection_name = getenv("LEMMA_COLLECTION")
    worker = EmbedWorker(lean_root_path, source_path, base_file, collection_name)
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Embed")
        raise e
    send_notification(False, job_name="Embed")
