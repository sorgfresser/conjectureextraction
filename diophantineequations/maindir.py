from argparse import ArgumentParser
from pathlib import Path

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
from dataclasses import dataclass
import logging
import json
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

@dataclass
class Conjecture:
    informal_problem: str
    informal_proof: str | None = None
    formal_problem: str | None = None
    formal_proof: str | None = None
    imports: list[str] | None = None
    parent_problem: str | None = None
    original_formalized: str | None = None

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("directory", help="Directory with the informal json files.")
    parser.add_argument("formaldirectory", help="Formal problem directory.")
    parser.add_argument("rootpath", help="Root path for the Lean project.")
    parser.add_argument("sourcepath", help="Source path to store the proven lemmas, must be a subdirectory of the root path.")
    parser.add_argument("basefile", help="Base file with the main conjecture to add the lemmas to.")
    parser.add_argument("sorrypath", help="Path to store the sorry conjectures, must be a subdirectory of the root path.")
    parser.add_argument("definitionpath", help="Path for definitions to use for autoformalization.")
    parser.add_argument("--prefix", help="Prefix for the formal conjecture, used for namespaces (e.g. open Lean)", default="")
    parser.add_argument("--problemid", help="Problem ID for the problems.", default="")
    parser.add_argument("--model", help="Model to use for conjecture generation.", default="o1-preview")
    parser.add_argument("--evaluate-conjectures", help="Evaluate the generated conjectures", action="store_true")
    parser.add_argument("--debug", help="Set to debug mode. Mostly influences logging for now", action="store_true")
    parser.add_argument("--formalization-path", help="Path to store informalized, formalization tuples for later use.", default=None)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("Debug mode enabled.")
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

    evaluate = args.evaluate_conjectures
    if evaluate:
        logger.info("Will evaluate the generated conjectures.")
    prefix = ""
    if args.prefix:
        prefix = args.prefix.replace("\\n", "\n")
        logger.info("Using prefix: %s", prefix)

    if args.problemid:
        logger.info("Problem ID: %s", args.problemid)
        lemma_vector_store = LemmaVectorStore.from_directory(src_path, ReProverEmbeddingFn(), args.problemid)  # use src path since root path would include the original IMO theorem
    else:
        lemma_vector_store = LemmaVectorStore.from_directory(src_path, ReProverEmbeddingFn())
    # File already exists, so we can skip the embedding and informalization
    #definition_vector_store = DefinitionVectorStore.from_directory(definition_path, DefaultEmbeddingFunction(), informalize)
    definition_vector_store = DefinitionVectorStore.from_file(DefaultEmbeddingFunction(), informalize)
    formalpath = Path(args.formaldirectory)
    conjecture_queue = Queue()
    conjecture_count = 0
    for file in Path(args.directory).rglob("*.json"):
        logger.info("Proving Problem: %s", file)
        with file.open("r") as f:
            problemargs = json.load(f)
        formalfile = formalpath / (file.stem + ".lean")
        if (not formalfile.exists()) or (not formalfile.is_file()):
            logger.warning("Formal file does not exist for %s, skipping.", file)
            continue
        with formalfile.open("r") as f:
            formalproblem = f.read()
        formalproblem = prefix + formalproblem if prefix else formalproblem
        logger.info("Full formal problem: %s", formalproblem)
        formalized = FormalizedConjecture(problemargs["problem_text"], formalproblem, [])
        conjecture_queue.put(Conjecture(problemargs["problem_text"], problemargs["solution_text"], formalproblem))
        conjecture_count += 1
    logger.info("Added %s conjectures to the queue.", conjecture_count)
    while not conjecture_queue.empty():
        conjecture: Conjecture = conjecture_queue.get()
        if conjecture.informal_proof is None:
            conjecture.informal_proof = sketch_proof(conjecture.formal_problem, conjecture.informal_problem, conjecture.parent_problem)
        conjectures = generate_conjectures(conjecture.informal_problem, conjecture.informal_proof, model=args.model)
        assert conjectures is not None
        # with open("o1_output.txt", "r") as file:
        #     conjectures_string = file.read()
        #     conjectures = ["**Reasoning" + e for e in conjectures_string.split("**Reasoning")]
        #     conjectures.pop(0)

        for conj in conjectures:
            logger.info("Processing conjecture: %s", conj)
            lean_files = definition_vector_store.get_definitions(conj, 10)
            formalized = formalize_conjecture(root_path, conj, lean_files, result_dir_path=formalization_path)
            if formalized is None:
                logger.warning("Could not formalize conjecture.")
                continue
            # Check doesnt exist
            if lemma_vector_store.exact_match(formalized.formalized_conjecture):
                logger.warning("Conjecture already exists in the source files.")
                continue

            reject = hypothesis_rejection(root_path, formalized, lemma_vector_store)
            if not reject:
                # Try to prove
                logger.info("Proving conjecture: %s", conj)
                proven, proven_theorem = prove_conjecture(root_path, formalized, lemma_vector_store)
                logger.info("Conjecture proven: %s", proven)
                logger.debug("Proof: %s", proven_theorem)
                if not proven:
                    logger.info("Putting conjecture %s in the queue.", conj)
                    conjecture_queue.put(Conjecture(conj, None, formalized.formalized_conjecture, None, formalized.imports, conjecture.informal_problem, conjecture.original_formalized if conjecture.original_formalized is not None else conjecture.formal_problem))
                    add_conjecture(root_path, sorry_path, base_file, formalized)
                else:
                    logger.info("Conjecture proven, adding to source files.")
                    file_path = add_conjecture(root_path, src_path, base_file, proven_theorem)
                    lemma_vector_store.add_single_file(file_path)
                    logger.debug("Added conjecture to the source file.")
                if evaluate:
                    logger.info("Evaluating conjecture.")
                    logger.debug(evaluate_conjecture(formalized.formalized_conjecture, conj, conjecture.original_formalized if conjecture.original_formalized is not None else conjecture.formal_problem))
        logger.info("Trying to prove again.")
        formalized = FormalizedConjecture(conjecture.informal_problem, conjecture.formal_problem, conjecture.imports if conjecture.imports is not None else [])
        proven, proven_theorem = prove_conjecture(root_path, formalized, lemma_vector_store, False)
        if not proven:
            logger.info("Could not prove the original conjecture.")
            conjecture_queue.put(conjecture)
        else:
            logger.info("Original conjecture proven, adding to source files.")
            file_path = add_conjecture(root_path, src_path, base_file, proven_theorem)
            lemma_vector_store.add_single_file(file_path)
            if not conjecture.parent_problem:
                logger.info("Original conjecture is the base conjecture, finalizing.")
                logger.warning("Final conjecture: %s", conjecture.formal_problem)
                logger.warning("Final proof: %s", conjecture.formal_proof)
                logger.warning("Imports: %s", conjecture.imports)
        logger.debug("Proof: %s", proven_theorem.proof if proven_theorem is not None else None)
