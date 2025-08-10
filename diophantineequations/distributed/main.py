from argparse import ArgumentParser
from pathlib import Path
from diophantineequations.distributed.workers.abstract import GenericConnector
from diophantineequations.utils import IS_DISTRIBUTED, RANK, conclusion_false
from diophantineequations.models import ProvenTheorem
from dataclasses import dataclass
import logging
# import weave
import wandb
import os
import json
import pika
from diophantineequations.distributed.messages import MasterToWorker, WorkerAction, ActionSearch, Conjecture, \
    WorkerToMaster, ResponseSearch, WorkerResponse, ResponseConjecture, ActionConjecture, RunConfig, ActionNoSearch, \
    AbstractSearchResponse, ResponseNoSearch, ActionRetrieve, ActionEmbed, ResponseEmbed, ResponseRetrieve
from diophantineequations.notify import send_notification

USE_WANDB = os.getenv("WANDB_API_KEY") is not None


@dataclass
class Stats:
    proven: int = 0
    rejected: int = 0
    total_proven: int = 0
    total_rejected: int = 0
    formalized: int = 0  # Successful formalizations (counts as one 1 for all up to k attempts)
    total_formalized: int = 0  # Total formalizations
    successful_formalization_attempts: int = 0  # Amount of attempts for successful
    total_formalization_attempts: int = 0  # Total formalization attempts
    roots_proven: int = 0
    total_roots_proven: int = 0
    total_expansions: int = 0
    search_time_total_ms: int = 0

    def sync(self, run: wandb.sdk.wandb_run.Run):
        api = wandb.Api()
        history = api.run(run.path).history(samples=100_000)
        if not history:
            return
        last = history[-1]
        self.update(last)
        self.roots_proven = last.get("root_theorems_proven", 0)
        self.total_roots_proven = last.get("total_root_theorems", 0)
        self.successful_formalization_attempts = last.get("formalized_attempts", 0)
        self.total_formalization_attempts = last.get("formalized_attempts_total", 0)
        self.search_time_total_ms = int(last.get("search_time_total", 0) * 1000)

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def inc_total_proven(self):
        self.total_proven += 1

    def inc_proven(self):
        self.proven += 1

    def inc_rejected(self):
        self.rejected += 1

    def inc_total_rejected(self):
        self.total_rejected += 1

    def _build_json(self):
        return {"proven": self.proven, "total_proven": self.total_proven,
                "proven_rate": float(self.proven) / self.total_proven if self.total_proven else 0.0,
                "rejected": self.rejected, "total_rejected": self.total_rejected,
                "rejected_rate": float(self.rejected) / self.total_rejected if self.total_rejected else 0.0,
                "formalized": self.formalized, "total_formalized": self.total_formalized,
                "formalized_rate": float(self.formalized) / self.total_formalized if self.total_formalized else 0.0,
                "root_theorems_proven": self.roots_proven, "total_root_theorems": self.total_roots_proven,
                "root_proven_rate": float(
                    self.roots_proven) / self.total_roots_proven if self.total_roots_proven else 0.0,
                "formalized_attempts": self.successful_formalization_attempts,
                "formalized_attempts_total": self.total_formalization_attempts, "formalized_attempts_avg": float(
                self.total_formalization_attempts) / self.successful_formalization_attempts if self.successful_formalization_attempts else 10.0,
                "total_expansions": self.total_expansions, "search_time_total": float(self.search_time_total_ms) / 1000,
                "avg_search_time": float(
                    self.search_time_total_ms) / 1000 / self.total_proven if self.total_proven else 0.0}

    def log(self):
        if USE_WANDB:
            wandb.log(self._build_json(), step=self.total_proven + self.total_rejected + self.total_formalized)


logger = logging.getLogger(__name__)


def check_args(args):
    if args.debug:
        if IS_DISTRIBUTED:
            logging.basicConfig(format=f"%(asctime)s %(levelname)s:Rank {RANK}:%(name)s:%(message)s",
                                level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            logging.basicConfig(level=logging.DEBUG)
        logger.info("Debug mode enabled.")
    else:
        if IS_DISTRIBUTED:
            logging.basicConfig(format=f"%(asctime)s %(levelname)s:Rank {RANK}:%(name)s:%(message)s",
                                level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            logging.basicConfig(level=logging.INFO)

    informal_dir = Path(args.directory)
    assert informal_dir.exists()
    formal_dir = Path(args.formaldirectory)
    assert formal_dir.exists()
    return informal_dir, formal_dir


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("directory", help="Directory with the informal json files.")
    parser.add_argument("formaldirectory", help="Formal problem directory.")
    parser.add_argument("search_queue", help="Name of the search queue in Rabbit MQ.")
    parser.add_argument("search_return_queue", help="Name of the return for search queue in Rabbit MQ.")
    parser.add_argument("conjecture_queue", help="Name of the conjecture queue in Rabbit MQ.")
    parser.add_argument("retrieval_queue", help="Name of the retrieval queue in Rabbit MQ.")
    parser.add_argument("embed_queue", help="Name of the embed queue in Rabbit MQ.")
    parser.add_argument("--debug", help="Set to debug mode. Mostly influences logging for now", action="store_true")
    parser.add_argument("--formalization-path", help="Path to store informalized, formalization tuples for later use.",
                        default=None)
    parser.add_argument("--expansions", type=int, help="Number of nodes to explore for the proof.", default=100)
    parser.add_argument("--expansions-rejection", type=int,
                        help="Number of nodes to explore for the hypothesis rejection.", default=100)
    parser.add_argument("--tactics-per-expansion", type=int, help="Number of tactis to generate per expansion",
                        default=32)
    parser.add_argument("--no-retrieval", action="store_true", help="Disable retrieval for formalization")
    parser.add_argument("--no-search", action="store_true", help="Disable search mechanisms.")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only using a single trivial statement.")
    parser.add_argument("--recursive", action="store_true", help="Whether to create new conjectures recursively.")
    parser.add_argument("--restart", action="store_true",
                        help="If set, will not enqueue new theorems but continue with existing elements in queue")
    parser.add_argument("--run-id", type=str, help="WandB Run ID to resume.")
    parser.add_argument("--attempts", type=int, help="Max conjecture extraction attempts per parent conjecture.", default=1_000_000) # basically inf
    parser.add_argument("--sequential", action="store_true", help="Only reenqueue a parent conjecture if child conjectures have been generated", default=False)
    return parser


class Master(GenericConnector):
    def __init__(self, search_queue_name: str, search_result_queue_name: str, conjecture_queue_name,
                 informal_dir: Path, formal_dir: Path, expansions: int, rejection_expansions: int,
                 num_tactics: int, retrieval_queue_name: str, embed_queue_name: str,
                 no_retrieval: bool, dry_run: bool, no_search: bool, recursive: bool, wandb_run: wandb.sdk.wandb_run.Run,
                 attempts: int, sequential: bool):
        super().__init__()
        self.channel.basic_qos(prefetch_count=1)
        self.channel.confirm_delivery()
        # Search
        self.search_reply_to = search_result_queue_name
        self.channel.queue_declare(queue=search_result_queue_name, auto_delete=False, durable=True)
        self.channel.basic_consume(self.search_reply_to, on_message_callback=self._search_result, auto_ack=False)
        self.search_queue = search_queue_name
        args = {
            'x-consumer-timeout': 36_000_000  # 36000 seconds, i.e. 10 hours
        }
        self.channel.queue_declare(queue=self.search_queue, auto_delete=False, durable=True, arguments=args)
        # Conjecture
        result = self.channel.queue_declare(queue="", exclusive=True)
        self.conjecture_reply_to = result.method.queue
        self.channel.basic_consume(self.conjecture_reply_to, on_message_callback=self._conjecture_callback,
                                   auto_ack=False)
        self.conjecture_queue = conjecture_queue_name
        args = {
            'x-consumer-timeout': 604_800_000  # 1 week
        }
        self.channel.queue_declare(queue=self.conjecture_queue, auto_delete=False, durable=True, arguments=args)
        # Embed
        result = self.channel.queue_declare(queue="", exclusive=True)
        self.embed_reply_to = result.method.queue
        self.channel.basic_consume(self.embed_reply_to, on_message_callback=self._embed_callback, auto_ack=False)
        self.embed_queue = embed_queue_name
        self.channel.queue_declare(queue=self.embed_queue, auto_delete=False, durable=True)
        # Retrieve
        result = self.channel.queue_declare(queue="", exclusive=True)
        self.retrieve_reply_to = result.method.queue
        self.channel.basic_consume(self.retrieve_reply_to, on_message_callback=self._retrieve_callback, auto_ack=False)
        self.retrieve_queue = retrieval_queue_name
        self.channel.queue_declare(queue=self.retrieve_queue, auto_delete=False, durable=True)
        # Internals
        self._informal_dir = informal_dir
        self._formal_dir = formal_dir
        self._num_tactics = num_tactics
        self._num_expansions = expansions
        self._num_rejection_expansions = rejection_expansions
        self._search_count = 0
        self.stats = Stats()
        self.stats.sync(wandb_run)
        self._run_config = RunConfig(retrieval=not no_retrieval, dry_run=dry_run, search=not no_search,
                                     recursive=recursive)
        self.attempts = attempts
        self.sequential = sequential

    def _init_fill_queue(self):
        assert self._search_count == 0
        if self._run_config.dry_run:
            logger.info("Filling dry run!")
            conjecture = Conjecture(informal_problem="Assuming h: False, it follows that False",
                                    informal_proof="This holds trivially using exact h",
                                    formal_problem="theorem triviality (h : False) : False := by     sorry")
            self._enqueue(conjecture)
            self.stats.total_roots_proven += 1
            return
        for file in self._informal_dir.rglob("*.json"):
            logger.info("Proving Problem: %s", file)
            with file.open("r") as f:
                problemargs = json.load(f)
            formalfile = self._formal_dir / (file.stem + ".lean")
            # Putnam fallback
            if not formalfile.exists():
                formalfile = self._formal_dir / (file.stem + "_sol.lean")
            if (not formalfile.exists()) or (not formalfile.is_file()):
                logger.warning("Formal file does not exist for %s, skipping.", file)
                continue
            with formalfile.open("r") as f:
                formalproblem = f.read()
            logger.info("Full formal problem: %s", formalproblem)
            # TODO: Premises and namespaces + imports again
            # Remove imports
            formalproblem_lines = [line.strip() for line in formalproblem.split("\n") if
                                   not line.strip().startswith("import") and line.strip()]
            formalproblem = "\n".join(formalproblem_lines)
            conjecture = Conjecture(informal_problem=problemargs["problem_text"],
                                    informal_proof=problemargs["solution_text"],
                                    formal_problem=formalproblem)
            self._enqueue(conjecture)
            self.stats.total_roots_proven += 1
        logger.info("Added %s conjectures to the queue.", self._search_count)

    def _enqueue(self, conjecture: Conjecture, premises: list[str] | None = None):
        premises = premises or []
        if self._run_config.search:
            action = ActionSearch(search_idx=self._search_count, theorem=conjecture.formal_problem,
                                  num_tactics=self._num_tactics, num_expansions=self._num_expansions,
                                  premises=premises, conjecture=conjecture, run_config=self._run_config)
        else:
            action = ActionNoSearch(search_idx=self._search_count, theorem=conjecture.formal_problem,
                                    num_tactics=self._num_tactics, max_tactics=5, # we're doing whole proof generation, so 5 should be fine
                                    premises=premises, conjecture=conjecture, run_config=self._run_config)
        msg = MasterToWorker(message=WorkerAction(action=action))
        self._search_count += 1
        self._send_search(msg)

    def _search_result(self, ch, method, props, body):
        from_worker = WorkerToMaster.model_validate_json(body)
        message = from_worker.message
        assert isinstance(message, WorkerResponse)
        response = message.response
        assert isinstance(response, AbstractSearchResponse)
        if response.error is not None:
            logger.error("Search failed with %s", response.error)
            # Discard in that case
            if "No theorem or lemma found!" == response.error:
                logger.error("Apparently search does not contain a lemma or theorem: %s", response.action.theorem)
                send_notification(True, f"No theorem or lemma found!\n{response.action}", "Main")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return None
        if response.action.is_inverted:
            self._handle_rejection(response)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return None
        # If proven, simply log
        self.stats.inc_total_proven()
        if isinstance(response, ResponseSearch):
            self.stats.total_expansions += response.expansions
        elif isinstance(response, ResponseNoSearch):
            self.stats.total_expansions += response.depth
        else:
            raise TypeError("Unsupported response type!")
        self.stats.search_time_total_ms += response.search_time_ms
        if response.proof:
            self.stats.inc_proven()
            if response.action.conjecture.parent_problem is None:
                self.stats.roots_proven += 1
            self.stats.log()
            self._handle_proven(response)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return None
        self.stats.log()

        # If not proven, decompose and try again after decompose
        if response.action.conjecture.attempts > self.attempts:
            logger.info("Attempts have been exceeded for conjecture, discarding!")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return None
        if response.action.conjecture.parent_problem is None:
            logger.info("Generating conjectures for a root theorem!")
            self._handle_conjecture(response)
        elif self._run_config.recursive:
            logger.info("Generating conjectures recursively!")
            self._handle_conjecture(response)
        if not self.sequential:
            logger.info("Not sequential, calling retrieve immediately!")
            msg = MasterToWorker(message=WorkerAction(action=ActionRetrieve(
                theorem=response.action.conjecture.formal_problem,
                past_tactics=[],
                goal=response.action.conjecture.formal_problem,
                k=10,
                search_idx=self._search_count,
                run_config=self._run_config,
                conjecture=response.action.conjecture,
            )))
            self._send_to_retrieve(msg)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return None

    def _handle_proven(self, response: AbstractSearchResponse):
        assert response.proof is not None
        imports = response.action.conjecture.imports if response.action.conjecture.imports else []
        proven_theorem = ProvenTheorem(
            nl_conjecture=response.action.conjecture.informal_problem,
            formalized_conjecture=response.action.theorem,
            imports=imports,
            proof=response.proof)
        msg = MasterToWorker(
            message=WorkerAction(action=ActionEmbed(search_idx=self._search_count, theorem=proven_theorem,
                                                    run_config=self._run_config, conjecture=response.action.conjecture,)), )
        self._send_to_embed(msg)

    def _handle_rejection(self, response: AbstractSearchResponse):
        logger.info("Handling hypothesis rejection %s", response.action)
        assert response.action.is_inverted
        # If we managed to prove, the conjecture was invalid and this strang ends
        self.stats.inc_total_rejected()
        if response.proof:
            logger.info("Conjecture was rejected, proof: %s", response.proof)
            self.stats.inc_rejected()
            self.stats.log()
            return
        self.stats.log()
        logger.info("Conjecture not rejected, retrieving...")
        # Not rejected, so we start retrieval before putting back in queue
        msg = MasterToWorker(message=WorkerAction(action=ActionRetrieve(
            theorem=response.action.conjecture.formal_problem,
            past_tactics=[],
            goal=response.action.conjecture.formal_problem,
            k=10,
            search_idx=self._search_count,
            run_config=self._run_config,
            conjecture=response.action.conjecture,
        )))
        self._send_to_retrieve(msg)

    def _handle_conjecture(self, response: AbstractSearchResponse):
        logger.info("Sending to conjecture generation")
        logger.debug("Formal problem being sent: %s", response.action.conjecture.formal_problem)
        msg = MasterToWorker(message=WorkerAction(
            action=ActionConjecture(conjecture=response.action.conjecture.model_dump(),
                                    definition_retrieval=self._run_config.retrieval,
                                    search_idx=self._search_count, run_config=self._run_config).model_dump()))
        self._send_conjecture(msg)

    def _conjecture_callback(self, ch, method, props, body):
        logger.info("Received conjecture callback!")
        from_worker = WorkerToMaster.model_validate_json(body)
        message = from_worker.message
        assert isinstance(message, WorkerResponse)
        response = message.response
        assert isinstance(response, ResponseConjecture)
        for idx, conjecture in enumerate(response.conjectures):
            self.stats.total_formalized += 1
            self.stats.total_formalization_attempts += response.attempts[idx]
            if conjecture.formal_problem is None:
                continue
            logger.info("Adding conjecture to queue...")
            logger.debug("Conjecture to be added: %s", conjecture.formal_problem)
            self.stats.formalized += 1
            self.stats.successful_formalization_attempts += response.attempts[idx]
            # Enqueue hypothesis rejection
            if self._run_config.search:
                action = ActionSearch(search_idx=self._search_count,
                                      theorem=conclusion_false(conjecture.formal_problem),
                                      num_expansions=self._num_expansions, premises=[], num_tactics=self._num_tactics,
                                      conjecture=conjecture, run_config=self._run_config, is_inverted=True)
            else:
                action = ActionNoSearch(search_idx=self._search_count,
                                        theorem=conclusion_false(conjecture.formal_problem),
                                        num_tactics=self._num_tactics, max_tactics=5,
                                        premises=[], conjecture=conjecture, run_config=self._run_config,
                                        is_inverted=True)
            msg = MasterToWorker(message=WorkerAction(action=action))
            self._search_count += 1
            self._send_search(msg)
        if self.sequential:
            logger.info("Sequential, calling retrieve now that conjectures are in search queue!")
            msg = MasterToWorker(message=WorkerAction(action=ActionRetrieve(
                theorem=response.action.conjecture.formal_problem,
                past_tactics=[],
                goal=response.action.conjecture.formal_problem,
                k=10,
                search_idx=self._search_count,
                run_config=self._run_config,
                conjecture=response.action.conjecture,
            )))
            self._send_to_retrieve(msg)
        self.stats.log()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _embed_callback(self, ch, method, props, body):
        logger.info("Received embed callback!")
        from_worker = WorkerToMaster.model_validate_json(body)
        message = from_worker.message
        assert isinstance(message, WorkerResponse)
        response = message.response
        assert isinstance(response, ResponseEmbed)
        if not response.success:
            logger.error("Failed to embed response!")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _retrieve_callback(self, ch, method, props, body):
        logger.info("Received retrieve callback!")
        from_worker = WorkerToMaster.model_validate_json(body)
        message = from_worker.message
        assert isinstance(message, WorkerResponse)
        response = message.response
        assert isinstance(response, ResponseRetrieve)
        assert all(file.full_proof is not None for file in response.files)
        conjecture = response.action.conjecture.model_copy(update={"attempts": response.action.conjecture.attempts + 1}, deep=True)
        self._enqueue(conjecture, [file.full_proof for file in response.files])
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _send_search(self, msg: MasterToWorker):
        self.channel.basic_publish(exchange="", routing_key=self.search_queue, body=msg.model_dump_json(),
                                   properties=pika.BasicProperties(reply_to=self.search_reply_to,
                                                                   delivery_mode=2))

    def _send_conjecture(self, msg: MasterToWorker):
        self.channel.basic_publish(exchange="", routing_key=self.conjecture_queue, body=msg.model_dump_json(),
                                   properties=pika.BasicProperties(reply_to=self.conjecture_reply_to, delivery_mode=2))

    def _send_to_embed(self, msg: MasterToWorker):
        self.channel.basic_publish(exchange="", routing_key=self.embed_queue, body=msg.model_dump_json(),
                                   properties=pika.BasicProperties(reply_to=self.embed_reply_to, delivery_mode=2))

    def _send_to_retrieve(self, msg: MasterToWorker):
        self.channel.basic_publish(exchange="", routing_key=self.retrieve_queue, body=msg.model_dump_json(),
                                   properties=pika.BasicProperties(reply_to=self.retrieve_reply_to, delivery_mode=2))

    def start(self, restart: bool = False):
        logger.info("Starting worker %s", self.__class__)
        if not restart:
            self._init_fill_queue()
        self.channel.start_consuming()


def main(args):
    informal_dir, formal_dir = check_args(args)
    logger.info("Informal problem directory %s", informal_dir)
    logger.info("Formal problem directory %s", formal_dir)
    if USE_WANDB:
        logger.info("Logging to W&B")
    master = Master(
        search_queue_name=args.search_queue, search_result_queue_name=args.search_return_queue,
        informal_dir=informal_dir, formal_dir=formal_dir,
        expansions=args.expansions, rejection_expansions=args.expansions_rejection,
        num_tactics=args.tactics_per_expansion, conjecture_queue_name=args.conjecture_queue,
        retrieval_queue_name=args.retrieval_queue, embed_queue_name=args.embed_queue, no_retrieval=args.no_retrieval,
        dry_run=args.dry_run, no_search=args.no_search, recursive=args.recursive, wandb_run=args.wandb_run,
        attempts=args.attempts, sequential=args.sequential
    )
    master.start(args.restart)


if __name__ == '__main__':

    import traceback

    try:
        parser = get_parser()
        params = parser.parse_args()
        if USE_WANDB:
            config = {"natural_language_directory": params.directory, "formal_directory": params.formaldirectory,
                      "debug": params.debug, "expansions": params.expansions,
                      "expansions_rejection": params.expansions_rejection,
                      "tactics_per_expansion": params.tactics_per_expansion,
                      "definition_retrieval": not params.no_retrieval, "dry_run": params.dry_run,
                      "no_search": params.no_search, "recursive": params.recursive,
                      "attempts": params.attempts, "sequential": params.sequential}
            # weave.init("diophantine")
            if params.run_id:
                run = wandb.init(project="diophantine", config=config, resume="must", id=params.run_id)
            else:
                run = wandb.init(project="diophantine", config=config)
            # with weave.attributes(config):
            params.wandb_run = run
            main(params)
        else:
            main(params)
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Main")
        raise e
    send_notification(False, job_name="Main")
