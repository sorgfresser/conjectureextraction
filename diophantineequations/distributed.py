from mpi4py import MPI
from mpi4py.MPI import Intracomm
from logging import getLogger
import logging
from diophantineequations.lemma_prove import prove_conjecture
from diophantineequations.lemma_embeddings import LemmaVectorStore, ReProverEmbeddingFn, CHROMA_HOST, CHROMA_PORT
from diophantineequations.definition_retrieval import informalize, DefinitionVectorStore, CHROMA_HOST, CHROMA_PORT
from diophantineequations.main import get_parser, check_args, Conjecture, generate_conjectures, add_conjecture, \
    sketch_proof, process_single
from diophantineequations.distributed_models import *
from diophantineequations.utils import RANK
from diophantineequations.reprover import get_model as get_model_reprover, generate_tactic as generate_reprover
from diophantineequations.deepseekprover import get_model as get_model_deepseek, generate_tactic as generate_deepseek
from queue import Queue
from pydantic import ValidationError
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import sys
from typing import Dict
import torch
sys.setrecursionlimit(10_000)

logger = getLogger(__name__)


def _receive_message(comm: Intracomm, rank: int) -> WorkerMessage:
    message = comm.recv(source=0)
    try:
        message = WorkerMessage.model_validate(message)
    except ValidationError as e:
        logger.error("Rank %s received invalid message: %s with errors: %s", rank, e, e.errors())
        logger.error("Message: %s", message)
        raise e
    return message

def _trainer_receive_message(comm: Intracomm, rank: int) -> TrainerMessage:
    message = comm.recv(source=0)
    try:
        message = TrainerMessage.model_validate(message)
    except ValidationError as e:
        logger.error("Rank %s received invalid message: %s", rank, e.errors())
        logger.error("Message: %s", message)
        raise e
    return message


def _reload_worker(model_path: Path):
    pass

def train_model(model_type: ModelType):
    if model_type == ModelType.reprover:
        pass
    pass


def train_worker(comm: Intracomm):
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s:Rank {RANK}:%(name)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    rank = comm.Get_rank()
    logger.info("Rank %s waiting for init", rank)
    init_request = WorkerRequest(request=WorkerRequestInit(worker_type=WorkerType.trainer))
    comm.send(init_request.model_dump(), dest=0)
    message = _trainer_receive_message(comm, rank)
    if not isinstance(message.action, ActionInit):
        logger.error("Rank %s received invalid message: %s", rank, message)
        raise ValueError("Invalid message")
    model_type = message.action.model_type
    model = get_model_reprover() if model_type == ModelType.reprover else get_model_deepseek()
    logger.info("Rank %s initializing", rank)

    ready_msg = WorkerResponse(response=WorkerReady())
    comm.send(ready_msg.model_dump(), dest=0)
    while True:
        has_trained = False
        message = _trainer_receive_message(comm, rank)
        action = message.action
        if isinstance(action, ActionStop):
            logger.warning("Rank %s received stop message, stopping", rank)
            break
        logger.info("Rank %s received message %s", rank, message)
        if isinstance(action, ActionTrain):
            logger.info("Rank %s training, saving to %s with model type %s and dataset path %s", rank, action.model_path, action.model_type, action.dataset_path)
            model_type = action.model_type
            model = get_model_reprover() if model_type == ModelType.reprover else get_model_deepseek()
            train_model()
            response = TrainedResponse(model_path=action.model_path, model_type=model_type, dataset_path=action.dataset_path)
            has_trained = True
        else:
            raise ValueError("Unknown action %s", action)
        response_msg = TrainerResponse(response=response)
        logger.info("Rank %s sending response %s", rank, response_msg)
        comm.send(response_msg.model_dump(), dest=0)
        if has_trained:
            for layer in model.state_dict().values():
                comm.Send(layer, dest=0)


def worker(comm: Intracomm):
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s:Rank {RANK}:%(name)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    rank = comm.Get_rank()
    logger.info("Rank %s waiting for init", rank)
    init_request = WorkerRequest(request=WorkerRequestInit(worker_type=WorkerType.environment))
    comm.send(init_request.model_dump(), dest=0)
    message = _receive_message(comm, rank)
    if not isinstance(message.action, ActionInit):
        logger.error("Rank %s received invalid message: %s", rank, message)
        raise ValueError("Invalid message")
    logger.info("Rank %s initializing", rank)
    vectorstore = LemmaVectorStore(ReProverEmbeddingFn(), collection_name=message.action.problem_id,
                                   hostname=CHROMA_HOST, port=CHROMA_PORT)
    model_type = message.action.model_type
    model = get_model_reprover() if model_type == ModelType.reprover else get_model_deepseek()
    logger.info("Rank %s initialized", rank)
    ready_msg = WorkerResponse(response=WorkerReady())
    comm.send(ready_msg.model_dump(), dest=0)
    while True:
        message = _receive_message(comm, rank)
        action = message.action
        if isinstance(action, ActionStop):
            logger.warning("Rank %s received stop message, stopping", rank)
            break
        logger.info("Rank %s received message %s", rank, message)
        if isinstance(action, ActionProve):
            logger.info("Rank %s proving conjecture %s", rank, action.conjecture)
            deepseek = model_type == ModelType.deepseek
            proven, proof = prove_conjecture(action.root_path, action.conjecture, vectorstore, deepseek)
            logger.info("Rank %s proven: %s", rank, proven)
            response = ProofResponse(proven=proven, proof=proof)
        elif isinstance(action, ActionTactics):
            logger.info("Rank %s getting tactics for goal %s", rank, action.goal)
            if model_type == ModelType.deepseek:
                tactics, scores = generate_deepseek(action.goal, action.premises, action.k)
            else:
                tactics, scores = generate_reprover(action.goal, action.premises, action.k)
            response = TacticsResponse(tactics=tactics, goal=action.goal, tactic_scores=scores.tolist())
        elif isinstance(action, ActionReload):
            model_type = action.model_type
            logger.info("Rank %s reloading from path %s and type %s", rank, action.model_path, model_type)
            _reload_worker(action.model_path)
            response = WorkerReady()
        else:
            raise ValueError("Unknown action %s", action)
        response_msg = WorkerResponse(response=response)
        logger.info("Rank %s sending response %s", rank, response_msg)
        comm.send(response_msg.model_dump(), dest=0)
    logger.info("Rank %s stopping", rank)


def master(comm, workers: int):
    parser = get_parser()
    args = parser.parse_args()
    root_path, src_path, base_file, definition_path, sorry_path, formalization_path, proof_path = check_args(args)
    assert args.problemid  # currently required
    worker_types = []
    model_type = ModelType.reprover if not args.deepseek else ModelType.deepseek

    logger.info("Rank 0 waiting for %s workers to initialize", workers)
    for idx in range(1, workers + 1):
        message = comm.recv(source=idx)
        try:
            message = WorkerRequest.model_validate(message)
        except ValidationError as e:
            logger.error("Rank 0 received invalid message from %s: %s with errors: %s", idx, e, e.errors())
            logger.error("Message: %s", message)
            raise e
        assert isinstance(message.request, WorkerRequestInit)
        worker_types.append(message.request.worker_type)
        logger.info("Rank 0 received init message from %s", idx)
        worker_msg = WorkerMessage(action=ActionInit(problem_id=args.problemid, model_type=model_type))
        comm.send(worker_msg.model_dump(), dest=idx)
    for idx in range(1, workers + 1):
        message = comm.recv(source=idx)
        try:
            message = WorkerResponse.model_validate(message)
        except ValidationError as e:
            logger.error("Rank 0 received invalid message from %s: %s with errors: %s", idx, e, e.errors())
            logger.error("Message: %s", message)
            raise e
        assert isinstance(message.response, WorkerReady)
        logger.info("Rank 0 received ready message from %s", idx)
    logger.info("All workers initialized")

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

    definition_vector_store = DefinitionVectorStore.from_file(DefaultEmbeddingFunction(), informalize)
    logger.info("Proving Problem: %s", args.problem)
    formalized = FormalizedConjecture(args.problem, formalproblem, [])
    proven, proof = prove_conjecture(root_path, formalized, lemma_vector_store, distributed=True, comm=comm,
                                     num_tactics=args.tactics, proof_path=proof_path, worker_types=worker_types)
    if proven:
        logger.info("Conjecture already proven, adding to source files.")
        file_path = add_conjecture(root_path, src_path, base_file, proof)
        lemma_vector_store.add_single_file(file_path)
        return
    conjecture_queue = Queue()
    conjecture_queue.put(Conjecture(args.problem, args.proof, formalproblem))
    logger.info("Added conjecture to the queue.")
    while not conjecture_queue.empty():
        conjecture: Conjecture = conjecture_queue.get()
        if conjecture.informal_proof is None:
            conjecture.informal_proof = sketch_proof(conjecture.formal_problem, conjecture.informal_problem,
                                                     conjecture.parent_problem)
        conjectures = generate_conjectures(conjecture.informal_problem, conjecture.informal_proof, model=args.model)
        assert conjectures is not None
        for conj in conjectures:
            result = process_single(conj, lemma_vector_store, formalization_path, definition_vector_store, root_path,
                                    src_path,
                                    base_file, conjecture.informal_problem, sorry_path, evaluate, formalproblem,
                                    distributed=True, comm=comm, num_tactics=args.tactics,
                                    num_rejection_tactics=args.tactics_rejection, proof_path=proof_path, worker_types=worker_types)
            if result is None:
                logger.info("Could not process conjecture %s, skipping.", conj)
                continue
            if args.recursive:
                logger.info("Recursively adding conjecture %s to queue.", result)
                conjecture_queue.put(result)
        logger.info("Trying to prove again.")
        formalized = FormalizedConjecture(conjecture.informal_problem, conjecture.formal_problem,
                                          conjecture.imports if conjecture.imports is not None else [])
        proven, proven_theorem = prove_conjecture(root_path, formalized, lemma_vector_store, distributed=True,
                                                  comm=comm, num_tactics=args.tactics, proof_path=proof_path)
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
                break
        logger.debug("Proof: %s", proven_theorem.proof if proven_theorem is not None else None)

    for idx in range(1, workers + 1):
        message = WorkerMessage(action=ActionStop())
        comm.send(message.model_dump(), dest=idx)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0:
        worker(comm)
    else:
        assert rank == 0
        master(comm, size - 1)


if __name__ == "__main__":
    main()
