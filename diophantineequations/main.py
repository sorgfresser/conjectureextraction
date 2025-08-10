from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from diophantineequations.evaluate_conjectures import evaluate_conjecture
from diophantineequations.hypothesis_rejection import hypothesis_rejection
from queue import Queue
from diophantineequations.lemma_add import add_conjecture
from diophantineequations.lemma_embeddings import LemmaVectorStore, ReProverEmbeddingFn
from diophantineequations.lemma_form import formalize_conjecture
from diophantineequations.models import FormalizedConjecture
from diophantineequations.lemma_gen import generate_conjectures
from diophantineequations.lemma_sketch import sketch_proof
from diophantineequations.lemma_prove import prove_conjecture
from diophantineequations.definition_retrieval import DefinitionVectorStore, informalize
from diophantineequations.distributed_models import WorkerType
from diophantineequations.utils import IS_DISTRIBUTED, RANK
from dataclasses import dataclass
import logging
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import weave
import wandb
import os

USE_WANDB = os.getenv("WANDB_API_KEY") is not None


@dataclass
class Conjecture:
    informal_problem: str
    informal_proof: str | None = None
    formal_problem: str | None = None
    formal_proof: str | None = None
    imports: list[str] | None = None
    parent_problem: str | None = None

@dataclass
class Stats:
    proven: int = 0
    rejected: int = 0
    total_proven: int = 0
    total_rejected: int = 0

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
                "rejected": self.rejected, "total_rejected": self.total_rejected}

    def log(self):
        if USE_WANDB:
            wandb.log(self._build_json(), step=self.total_proven + self.total_rejected)


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

    root_path = Path(args.rootpath)
    assert root_path.exists()
    src_path = Path(args.sourcepath)
    base_file = Path(args.basefile)
    assert base_file.exists()
    definition_path = Path(args.definitionpath)
    assert definition_path.exists()
    assert definition_path.is_relative_to(root_path), "Definition path must be a subdirectory of the root path."
    sorry_path = Path(args.sorrypath)
    if not sorry_path.exists():
        logger.warning("Sorry path does not exist, creating.")
        sorry_path.mkdir()
    if not src_path.exists():
        logger.warning("Source path does not exist, creating.")
        src_path.mkdir()
    formalization_path = None
    if args.formalization_path:
        logger.info("Formalization path: %s", args.formalization_path)
        formalization_path = Path(args.formalization_path)
        if not formalization_path.exists():
            logger.warning("Formalization path does not exist, creating.")
            formalization_path.mkdir()
    proof_path = None
    if args.proof_path:
        logger.info("Proof path: %s", args.proof_path)
        proof_path = Path(args.proof_path)
        if not proof_path.exists():
            logger.warning("Proof path does not exist, creating.")
            proof_path.mkdir()
    return root_path, src_path, base_file, definition_path, sorry_path, formalization_path, proof_path


def process_single(conjecture: str, lemma_vector_store: LemmaVectorStore, formalization_path: Path,
                   definition_vector_store: DefinitionVectorStore, root_path: Path, src_path: Path, base_file: Path,
                   parent_informal: str, sorry_path: Path, evaluate: bool, root_formal: str,
                   json_path: Path, model_path: Path, stats: Stats,
                   distributed: bool = False, comm=None, num_tactics: int = 30, num_rejection_tactics: int = 30,
                   proof_path: Path | None = None, worker_types: list[WorkerType] | None = None, deepseek: bool = False) -> Conjecture | None:
    result = None
    logger.info("Processing conjecture: %s", conjecture)
    lean_files = definition_vector_store.get_definitions(conjecture, 10)
    formalized = formalize_conjecture(root_path, conjecture, lean_files, result_dir_path=formalization_path)
    if formalized is None:
        logger.warning("Could not formalize conjecture.")
        return None
    # Check doesnt exist
    if lemma_vector_store.exact_match(formalized.formalized_conjecture):
        logger.warning("Conjecture already exists in the source files.")
        return None
    hyp_json_path = json_path.with_stem(json_path.stem + "_hyp")
    hyp_model_path = model_path.with_stem(model_path.stem + "_hyp")
    reject = hypothesis_rejection(root_path, formalized, lemma_vector_store, hyp_json_path, hyp_model_path,
                                  distributed=distributed, comm=comm, num_tactics=num_rejection_tactics,
                                  proof_path=proof_path, worker_types=worker_types, deepseek=deepseek)
    stats.inc_total_rejected()
    if reject:
        stats.inc_rejected()
        stats.log()
        logger.warning("Rejected conjecture: %s", formalized)
        return None
    stats.log()
    # Try to prove
    logger.info("Proving conjecture: %s", conjecture)
    proven, proven_theorem = prove_conjecture(root_path, formalized, lemma_vector_store,
                                              json_path, model_path,
                                              distributed=distributed,
                                              comm=comm, num_tactics=num_tactics, proof_path=proof_path,
                                              deepseek=deepseek, worker_types=worker_types)
    logger.info("Conjecture proven: %s", proven)
    logger.debug("Proof: %s", proven_theorem)
    stats.inc_total_proven()
    if not proven:
        logger.info("Putting conjecture %s in the queue.", conjecture)
        result = Conjecture(conjecture, None, formalized.formalized_conjecture, None, formalized.imports,
                            parent_informal)
        add_conjecture(root_path, sorry_path, base_file, formalized)
    else:
        stats.inc_proven()
        logger.info("Conjecture proven, adding to source files.")
        file_path = add_conjecture(root_path, src_path, base_file, proven_theorem)
        if not file_path.exists():
            logger.warning("Could not add conjecture, skipping adding to vector store!")
        else:
            lemma_vector_store.add_single_file(file_path)
            logger.debug("Added conjecture to the source file.")
    stats.log()
    if evaluate:
        logger.info("Evaluating conjecture.")
        logger.debug(evaluate_conjecture(formalized.formalized_conjecture, conjecture, root_formal))
    return result


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("problem", help="Natural language problem statement.")
    parser.add_argument("proof", help="Natural language proof for the problem.")
    parser.add_argument("formalproblem", help="Formal problem statement.")
    parser.add_argument("rootpath", help="Root path for the Lean project.")
    parser.add_argument("sourcepath",
                        help="Source path to store the proven lemmas, must be a subdirectory of the root path.")
    parser.add_argument("basefile", help="Base file with the main conjecture to add the lemmas to.")
    parser.add_argument("sorrypath",
                        help="Path to store the sorry conjectures in, must be a subdirectory of the root path.")
    parser.add_argument("definitionpath", help="Path for definitions to use for autoformalization.")
    parser.add_argument("--prefix", help="Prefix for the formal conjecture, used for namespaces (e.g. open Lean)",
                        default="")
    parser.add_argument("--problemid", help="Problem ID for the problem.", default="")
    parser.add_argument("--model", help="Model to use for conjecture generation.", default="o1-preview")
    parser.add_argument("--evaluate-conjectures", help="Evaluate the generated conjectures", action="store_true")
    parser.add_argument("--debug", help="Set to debug mode. Mostly influences logging for now", action="store_true")
    parser.add_argument("--formalization-path", help="Path to store informalized, formalization tuples for later use.",
                        default=None)
    parser.add_argument("--proof-path", help="Path to store the proofs for later use.", default=None)
    parser.add_argument("--tactics", type=int, help="Number of tactics to explore for the proof.", default=100)
    parser.add_argument("--tactics-rejection", type=int, help="Number of tactics to explore for the rejection.", default=100)
    parser.add_argument("--recursive", action="store_true", help="Whether to recursively generate and prove conjectures.")
    parser.add_argument("--deepseek", action="store_true", help="Whether to use deepseek or reprover")
    parser.add_argument("--online", action="store_true", help="Whether to use online RL or not.")

    return parser


def main(args):
    root_path, src_path, base_file, definition_path, sorry_path, formalization_path, proof_path = check_args(args)
    stats = Stats()
    evaluate = args.evaluate_conjectures
    if evaluate:
        logger.info("Will evaluate the generated conjectures.")
    prefix = ""
    if args.prefix:
        prefix = args.prefix.replace("\\n", "\n")
        logger.info("Using prefix: %s", prefix)
    formalproblem = prefix + args.formalproblem if prefix else args.formalproblem
    logger.info("Full formal problem: %s", formalproblem)
    if args.problemid:
        logger.info("Problem ID: %s", args.problemid)
        lemma_vector_store = LemmaVectorStore.from_directory(src_path, ReProverEmbeddingFn(),
                                                             args.problemid)  # use src path since root path would include the original IMO theorem
    else:
        lemma_vector_store = LemmaVectorStore.from_directory(src_path, ReProverEmbeddingFn())
    logger.info("Logging to W&B")
    deepseek = args.deepseek
    logger.info("Deepseek set to %s", deepseek)
    # File already exists, so we can skip the embedding and informalization
    # definition_vector_store = DefinitionVectorStore.from_directory(definition_path, DefaultEmbeddingFunction(), informalize)
    definition_vector_store = DefinitionVectorStore.from_file(DefaultEmbeddingFunction(), informalize)
    logger.info("Proving Problem: %s", args.problem)

    formalized = FormalizedConjecture(args.problem, formalproblem, [])
    idx = 0
    if deepseek:
        json_path = Path(f"jsons_{args.problemid}_deepseek/")
    else:
        json_path = Path(f"jsons_{args.problemid}/")
    if deepseek:
        model_path = Path(f"{args.problemid}_{idx}_deepseek.pth")
    else:
        model_path = Path(f"{args.problemid}_{idx}.pth")
    proven, proof = prove_conjecture(root_path, formalized, lemma_vector_store, json_path, model_path, num_tactics=args.tactics, proof_path=proof_path, deepseek=deepseek)
    stats.inc_total_proven()
    if proven:
        stats.inc_proven()
        stats.log()
        logger.error("Statement was immediately proven, exiting")
        return
    idx += 1
    stats.log()
    conjecture_queue = Queue()
    conjecture_queue.put(Conjecture(args.problem, args.proof, formalproblem))

    while not conjecture_queue.empty():
        conjecture: Conjecture = conjecture_queue.get()
        if conjecture.informal_proof is None:
            conjecture.informal_proof = sketch_proof(conjecture.formal_problem, conjecture.informal_problem,
                                                     conjecture.parent_problem)
        conjectures = generate_conjectures(conjecture.informal_problem, conjecture.informal_proof, model=args.model)
        assert conjectures is not None
        # with open("o1_output.txt", "r") as file:
        #     conjectures_string = file.read()
        #     conjectures = ["**Reasoning" + e for e in conjectures_string.split("**Reasoning")]
        #     conjectures.pop(0)

        for conj in conjectures:
            if deepseek:
                model_path = Path(f"{args.problemid}_{idx}_deepseek.pth")
            else:
                model_path = Path(f"{args.problemid}_{idx}.pth")
            result = process_single(conj, lemma_vector_store, formalization_path, definition_vector_store, root_path,
                                    src_path, base_file, conjecture.informal_problem, sorry_path, evaluate,
                                    formalproblem, json_path, model_path, stats,
                                    num_tactics=args.tactics, num_rejection_tactics=args.tactics_rejection,
                                    proof_path=proof_path, deepseek=deepseek)
            idx += 1
            if result is None:
                logger.info("Could not process conjecture %s, skipping.", conj)
                continue
            if args.recursive:
                logger.info("Recursively adding conjecture %s to the queue.", conj)
                conjecture_queue.put(result)
        logger.info("Trying to prove again.")
        formalized = FormalizedConjecture(conjecture.informal_problem, conjecture.formal_problem,
                                          conjecture.imports if conjecture.imports is not None else [])
        if deepseek:
            model_path = Path(f"{args.problemid}_{idx}_deepseek.pth")
        else:
            model_path = Path(f"{args.problemid}_{idx}.pth")
        proven, proven_theorem = prove_conjecture(root_path, formalized, lemma_vector_store, json_path, model_path,
                                                  deepseek=deepseek, num_tactics=args.tactics, proof_path=proof_path)
        stats.inc_total_proven()
        idx += 1
        if not proven:
            logger.info("Could not prove the original conjecture.")
            conjecture_queue.put(conjecture)
        else:
            stats.inc_proven()
            logger.info("Original conjecture proven, adding to source files.")
            file_path = add_conjecture(root_path, src_path, base_file, proven_theorem)
            lemma_vector_store.add_single_file(file_path)
            if not conjecture.parent_problem:
                stats.log()
                logger.info("Original conjecture is the base conjecture, finalizing.")
                logger.warning("Final conjecture: %s", conjecture.formal_problem)
                logger.warning("Final proof: %s", conjecture.formal_proof)
                logger.warning("Imports: %s", conjecture.imports)
                break
        logger.debug("Proof: %s", proven_theorem.proof if proven_theorem is not None else None)
        stats.log()


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    if USE_WANDB:
        config = {"problem": params.problem, "proof": params.proof, "formalproblem": params.formalproblem,
                               "prefix": params.prefix, "problem_id": params.problemid, "model": params.model,
                               "evaluate": params.evaluate_conjectures, "debug": params.debug,
                  "deepseek": params.deepseek, "online": params.online}
        weave.init("diophantine")
        wandb.init("diophantine", config=config)
        with weave.attributes(config):
            main(params)
    else:
        main(params)
