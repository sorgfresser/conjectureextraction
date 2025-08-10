from pathlib import Path
import json
import torch.nn.functional
import time
import logging
from functools import partial
from pydantic import ValidationError, BaseModel
from typing import Optional, Tuple, Union, Callable
from lean_repl_py import LeanREPLHandler, LeanREPLProofState, LeanREPLNextProofState, LeanREPLEnvironment
from htps import HTPS, Theorem, Context, SearchParams, PolicyType, QValueSolved, NodeMask, Metric, EnvEffect, EnvExpansion, Tactic, Proof, SampleTactics
from diophantineequations.lemma_embeddings import LemmaVectorStore
from diophantineequations.utils import text_without_comments, theorem_name_span, replace_sorry
from diophantineequations.models import FormalizedConjecture, ProvenTheorem
from diophantineequations.deepseekprover import get_model, generate_tactic as ds_generate_tactic, train as train_ds, DEFAULT_MODEL
from diophantineequations.reprover import generate_tactic, train
from diophantineequations.environment import get_handler
from diophantineequations.environment import import_code
import anthropic
import heapq
import weave
from diophantineequations.distributed_models import ActionTactics, TacticsResponse, WorkerMessage, WorkerResponse, WorkerType
from dataclasses import dataclass, field

client = anthropic.Anthropic()

# client = OpenAI()
logger = logging.getLogger(__name__)

MAX_TRIES = 64
K_PREMISES = 10

BLACKLISTED = ["sorry", "apply sorry"]



class ProofSample(BaseModel):
    goal: str
    premises: list[str]
    proof: str
    all_goals: list[str]
    tactic: str

    def save_json(self, path: Path):
        path.write_text(self.model_dump_json())



@dataclass
class Node:
    goal: str
    all_goals: list[str]
    parent: Union["Node", None]
    children_for_tactic: dict[str, list["Node"]]
    solved: bool = False
    proof: str = ""
    parent_tactic: str = ""
    solving_tactic: str = ""
    repl_idx: int = 0
    # Per node, as we have to allow seeing the same goal in siblings (because different tactics, so different proofs)
    goals_seen: set[str] = field(default_factory=set)

    def backup_solved(self, tactic: str):
        if (not self.solved) and all(child.solved for child in self.children_for_tactic[tactic]):
            self.proof = tactic + "\n" + "\n".join([child.proof for child in self.children_for_tactic[tactic]])
            self.solved = True
            self.solving_tactic = tactic
            if self.parent:
                self.parent.backup_solved(self.parent_tactic)

    def get_proofs(self, premises: list[str] | None = None) -> list[ProofSample]:
        if premises is None:
            premises = []
        proofs = []
        goals = set()
        goals.add(self.goal)
        if self.solved:
            proofs.append(ProofSample(goal=self.goal, premises=premises, proof=self.proof, tactic=self.solving_tactic,
                                      all_goals=self.all_goals))
        for children in self.children_for_tactic.values():
            for child in children:
                for proof in child.get_proofs(premises):
                    if proof.goal not in goals:
                        proofs.append(proof)
                        goals.add(proof.goal)
        return proofs


@weave.op()
def prove_distributed(handler: LeanREPLHandler, proof_state: LeanREPLProofState, premises: list[str],
                      comm, num_tactics: int = 30, get_proofs: bool = False) -> Tuple[
    bool, Optional[str], list[ProofSample]]:
    workers = comm.Get_size() - 1
    worker_idle = [True] * workers
    goals_seen = set()
    goals_seen.add(proof_state.goal.strip())
    root = Node(proof_state.goal, [proof_state.goal], None, {}, repl_idx=proof_state.proof_state,
                goals_seen={proof_state.goal.strip()})
    goals_queue = []
    heapq.heappush(goals_queue, (0.0, proof_state.goal.strip()))
    nodes: dict[str, Node] = {proof_state.goal.strip(): root}
    logger.info("Starting distributed proof")
    logger.debug("Premises: %s", premises)
    logger.debug("Nodes: %s", nodes)
    time_last = time.time()
    while len(nodes) < num_tactics:
        if root.solved:
            break
        if not goals_queue and all(worker_idle):
            logger.info("Queue empty and all workers idle, breaking")
            break
        # Receive non-blocking here
        for i in range(1, workers + 1):
            if comm.Iprobe(source=i):
                worker_response = comm.recv(source=i)
                logger.info("Received response %s from worker %s", worker_response, i)
                try:
                    worker_response = WorkerResponse.model_validate(worker_response)
                except ValidationError as e:
                    logger.error("Received invalid message from worker %s: %s", i, e)
                    logger.error("Message: %s", worker_response)
                    raise e
                response = worker_response.response
                if isinstance(response, TacticsResponse):
                    node = nodes[response.goal]
                    if node.solved:
                        continue
                    for tactic, score in zip(response.tactics, response.tactic_scores):
                        if tactic.strip() in BLACKLISTED:
                            logger.debug("Skipping blacklisted tactic %s", tactic)
                            continue
                        if tactic.strip().startswith("have"):
                            logger.debug("Skipping have tactic")
                            continue
                        if tactic in node.children_for_tactic:
                            continue
                        handler.send_tactic(tactic, node.repl_idx)
                        response, env = handler.receive_json()
                        if isinstance(response, LeanREPLNextProofState):
                            # If error
                            if any(msg.severity == "error" for msg in response.messages):
                                continue
                            if not response.goals:
                                node.solved = True
                                node.proof = tactic
                                node.solving_tactic = tactic
                                if node.parent:
                                    node.parent.backup_solved(node.parent_tactic)
                                break
                            goal = response.goals[0]
                            if goal.strip() in node.goals_seen:
                                logger.info("Skipping tactic %s", tactic)
                                break
                            new_node = Node(goal=goal, all_goals=response.goals, parent=node, children_for_tactic={},
                                            goals_seen=node.goals_seen | {goal}, repl_idx=response.proof_state,
                                            parent_tactic=tactic)
                            goals_seen.add(new_node.goal)
                            if new_node.goal in nodes:
                                if tactic in node.children_for_tactic:
                                    if not any(
                                            child.goal == new_node.goal for child in node.children_for_tactic[tactic]):
                                        node.children_for_tactic[tactic].append(nodes[new_node.goal])
                                else:
                                    node.children_for_tactic[tactic] = [nodes[new_node.goal]]
                                continue
                            assert new_node.goal not in nodes
                            nodes[new_node.goal] = new_node
                            if tactic in node.children_for_tactic:
                                node.children_for_tactic[tactic].append(new_node)
                            else:
                                node.children_for_tactic[tactic] = [new_node]
                            heapq.heappush(goals_queue, (score, new_node.goal))
                        else:
                            logger.debug("Lean REPL failed, returned %s", response)
                    worker_idle[i - 1] = True
                    logger.debug("Node count: %s", len(nodes))
                else:
                    logger.error("Received unexpected message %s", response)
                    raise ValueError("Unexpected message")
        if not goals_queue:
            if time_last + 5 < time.time():
                logger.debug("Queue empty")
                time_last = time.time()
            continue
        for i in range(1, workers + 1):
            if not goals_queue:
                break
            if worker_idle[i - 1]:
                goal = heapq.heappop(goals_queue)[1]
                logger.info("Sending goal %s to worker %s", goal, i)
                worker_msg = WorkerMessage(action=ActionTactics(goal=goal, premises=premises, k=MAX_TRIES))
                comm.send(worker_msg.model_dump(), dest=i)
                worker_idle[i - 1] = False
    logger.info("Finishing proof, clearing workers...")
    for i in range(1, workers + 1):
        if not worker_idle[i - 1]:
            worker_response = comm.recv(source=i)
            logger.info("Received response %s from worker %s", worker_response, i)
            worker_idle[i - 1] = True
    logger.info("Finished distributed proof")
    logger.debug("Solved: %s", root.solved)
    logger.debug("Proof: %s", root.proof)
    proofs = root.get_proofs(premises) if get_proofs else []
    return root.solved, root.proof, proofs


def expand_theorem(handler: LeanREPLHandler, theorem: Theorem, premises: list[str], context: Context, data_dict,
                   generate_tactic_fn: Callable[[str, list[str], int], Tuple[list[str], torch.FloatTensor]],
                   deepseek: bool = False)\
        -> EnvExpansion:
    tactic_strs, scores = generate_tactic_fn(theorem.conclusion, premises, MAX_TRIES)
    solving_ids = []
    children_for_tactic = []
    tactics = []
    effects = []
    times = []
    worked_idx = 0
    valid_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)
    for idx, tactic_str in enumerate(tactic_strs):
        tactic_str = tactic_str.strip()
        if tactic_str in BLACKLISTED:
            continue
        start_time = time.time_ns()
        handler.send_tactic(tactic_str, theorem.metadata["proof_state_idx"])
        time_taken = time.time_ns() - start_time
        print("Tactic", tactic_str)
        try:
            output, env = handler.receive_json()
        except ValidationError as e:
            logger.exception("Validation error when receiving json")
            logger.error("Errors: %s", e.errors())
            continue
        if isinstance(output, LeanREPLNextProofState):
            # If error
            if any(msg.severity == "error" for msg in output.messages):
                continue
            # Worked
            tactic = Tactic(tactic_str, is_valid=True, duration=time_taken // 1_000_000)
            times.append(time_taken // 1_000_000)
            if deepseek:
                conclusion =  theorem.conclusion + "\n" + tactic_str
                if output.goals:
                    children = [Theorem( conclusion,  conclusion, [], context, theorem.past_tactics + [tactic])]
                else:
                    children = []
            else:
                children = [Theorem(goal, goal, [], context, theorem.past_tactics + [tactic]) for goal in output.goals]
            for child in children:
                child.metadata = {"proof_state_idx": output.proof_state}
                data_dict[child.conclusion] = output.proof_state
            children_for_tactic.append(children)
            effects.append(EnvEffect(theorem, tactic, children))
            tactics.append(tactic)
            valid_mask[idx] = True
            if not children:
                solving_ids.append(worked_idx)
            worked_idx += 1

    priors = scores[valid_mask]
    # In case we solve, only give solving effects
    if solving_ids:
        print("Solving with ids", solving_ids)
        print(tactics)
        tactics = [tactics[i] for i in solving_ids]
        print(tactics)
        priors = torch.tensor([priors[i] for i in solving_ids], device=scores.device)
        effects = [effects[i] for i in solving_ids]
        children_for_tactic = [children_for_tactic[i] for i in solving_ids]
        times = [times[i] for i in solving_ids]


    # Renormalize priors to 1
    if len(priors) > 0:
        priors = torch.nn.functional.softmax(priors, dim=-1)
        priors_list = priors.tolist()
        expansion = EnvExpansion(theorem, 1, 1, times, effects, -0.5, tactics=tactics,
                                 children_for_tactic=children_for_tactic, priors=priors_list)
    else:
        expansion = EnvExpansion(theorem, 1, 1, [], "The expansion failed!")
    return expansion

def get_proof(proof: Proof) -> list[str]:
    tactics = [proof.tactic.unique_string]
    if not proof.children:
        return tactics
    return tactics + get_proof(proof.children[0])

def expand_reprover(theorems: list[Theorem], handler: LeanREPLHandler, context: Context, thm_dict: dict[str, int], premises: list[str]) -> list[EnvExpansion]:
    expansions = []
    for theorem in theorems:
        expansions.append(expand_theorem(handler, theorem, premises, context, thm_dict, generate_tactic))
    return expansions

def expand_deepseek(theorems: list[Theorem], handler: LeanREPLHandler, context: Context, thm_dict: dict[str, int]) -> list[EnvExpansion]:
    expansions = []
    for theorem in theorems:
        expansions.append(expand_theorem(handler, theorem, [], context, thm_dict, ds_generate_tactic, deepseek=True))
    return expansions

class TacticSamplesDataset(torch.utils.data.Dataset):
    def __init__(self, json_dir: Path):
        self.json_dir = json_dir
        self.files = list(json_dir.glob("*.json"))
        self.file_contents = []
        for f in self.files:
            with f.open() as file:
                self.file_contents.append(file.read())
        self._deduplicate()

    def __len__(self):
        return len(self.file_contents)

    def _deduplicate(self):
        self.file_contents = list(dict.fromkeys(self.file_contents))

    def __getitem__(self, item):
        return json.loads(self.file_contents[item])


@weave.op()
def prove_htps(handler: LeanREPLHandler, proof_state: LeanREPLProofState, expansion_fn: Callable[[list[Theorem], LeanREPLHandler, Context, dict[str, int]], list[EnvExpansion]],
                train_fn: Callable[[list[SampleTactics]], Path], num_tactics: int = 30) -> Union[str, None]:
    context = Context([])
    root_thm = Theorem(proof_state.goal, unique_string=proof_state.goal, hypotheses=[], context=context, past_tactics=[])
    root_thm.metadata = {"proof_state_idx": proof_state.proof_state}
    params = SearchParams(0.3, PolicyType.RPO, num_expansions=num_tactics, succ_expansions=3,
                          early_stopping=None, no_critic=False, backup_once=False, backup_one_for_solved=True,
                          depth_penalty=0.99, count_threshold=10, tactic_p_threshold=True,
                          tactic_sample_q_conditioning=False, only_learn_best_tactics=False, tactic_init_value=0.0,
                          q_value_solved=QValueSolved.One, policy_temperature=0.7, metric=Metric.Size,
                          node_mask=NodeMask.MinimalProofSolving, effect_subsampling_rate=1.0, critic_subsampling_rate=1.0,
                          early_stopping_solved_if_root_not_proven=True, virtual_loss=0)
    search = HTPS(root_thm, params)
    idx = 0
    current_idx = 0
    thm_dict = {root_thm.conclusion: proof_state.proof_state}
    with open("samples/search.json", "w") as f:
        json.dump(json.loads(search.get_json_str()), f)
    while not search.is_done():
        print("Index", current_idx + 1)
        theorems: list[Theorem] = search.theorems_to_expand()
        for theorem in theorems:
            theorem.metadata = {"proof_state_idx": thm_dict[theorem.conclusion]}
            idx += 1
        expansions = expansion_fn(theorems, handler, context, thm_dict)
        expansion_dicts = [json.loads(expansion.get_json_str()) for expansion in expansions]
        current_idx += 1
        with open(f"samples/expansions_{current_idx}.json", "w") as f:
            json.dump(expansion_dicts, f)
        with open(f"samples/search_{current_idx}.json", "w") as f:
            json.dump(json.loads(search.get_json_str()), f)
        search.expand_and_backup(expansions)
        if idx % 200 == 0:
            result = search.get_result()
            train_fn(result.tactic_samples)

    result = search.get_result()
    if result.proof is None:
        return None
    train_fn(result.tactic_samples)
    tactics = get_proof(result.proof)
    return "\n".join(tactics)


@weave.op()
def _send_command(handler: LeanREPLHandler, command: str) -> tuple[
    dict[str, str | LeanREPLProofState] | LeanREPLNextProofState, LeanREPLEnvironment | None]:
    handler.send_command(command=command)
    response = handler.receive_json()
    assert response is not None
    output, env = response
    return output, env

def collate_fn(batch):
    return batch


def _create_dataset(tactic_samples: list[SampleTactics], json_dir: Path, premises: list[str]) -> TacticSamplesDataset:
    if not json_dir.exists():
        logger.warning("Creating json dir...")
        json_dir.mkdir(parents=True)
    indices = list(json_dir.glob("*.json"))
    highest_idx = max(int(name.stem) for name in indices) + 1 if indices else 0
    for idx, sample in enumerate(tactic_samples):
        with (json_dir / f"{highest_idx + idx}.json").open("w") as f:
            best_tactic_id = max(range(0, len(sample.tactics)), key=lambda x: sample.target_pi[x])
            json.dump({"premises": premises, "state": sample.goal.conclusion, "tactic": sample.tactics[best_tactic_id].unique_string}, f)
    dataset = TacticSamplesDataset(json_dir)
    return dataset

def train_reprover(tactic_samples: list[SampleTactics], json_dir: Path, model_path: Path,
                   premises: list[str], batch_size: int = 10) -> Path:
    dataset = _create_dataset(tactic_samples, json_dir, premises)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train(dataloader, model_path)

def train_deepseek(tactic_samples: list[SampleTactics], json_dir: Path, model_path: Path, premises: list[str], batch_size: int = 10) -> Path:
    dataset = _create_dataset(tactic_samples, json_dir, premises)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for sample in tactic_samples:
        print(sample.goal, sample.target_pi, [tactic.unique_string for tactic in sample.tactics])
    path = train_ds(dataloader, DEFAULT_MODEL, str(model_path))
    get_model(path)
    return Path(path)



@weave.op()
def prove_conjecture(root_path: Path, conjecture: FormalizedConjecture,
                     vector_store: LemmaVectorStore, json_path: Path,
                     model_path: Path, deepseek: bool = False,
                     distributed: bool = False, comm=None, num_tactics: int = 30,
                     proof_path: Optional[Path] = None, worker_types: Optional[list[WorkerType]] = None) -> Tuple[bool, Optional[ProvenTheorem]]:
    # if proof_path and not distributed:
    #     raise ValueError("Cannot get proofs without distributed mode")
    conjecture.formalized = text_without_comments(conjecture.formalized_conjecture)
    handler = get_handler(root_path, deepseek)
    full = "\n".join(conjecture.imports) + "\n" + conjecture.formalized_conjecture
    handler = import_code(handler, full)
    # Remove imports
    conjecture_lines = [line for line in conjecture.formalized_conjecture.split("\n") if
                        not line.strip().startswith("import")]
    conjecture_formalized = "\n".join(conjecture_lines)
    logger.info("Executing %s", conjecture_formalized)
    # Starting tactic mode
    output, env = _send_command(handler, conjecture_formalized)
    sorries = output["sorries"]
    assert len(sorries) == 1  # We expect one "by sorry" immediately after the conjecture
    state: LeanREPLProofState = sorries[0]
    import_strings = None
    logger.info("Proving without distributed tactic generation")
    if deepseek:
        get_model()
        deepseek_expansion_fn = expand_deepseek
        theorem_without_sorry = replace_sorry(conjecture_formalized, "")
        print("Formalised conjecture", theorem_without_sorry)
        state.goal = theorem_without_sorry
        deepseek_train_fn = partial(train_deepseek,json_dir=json_path, model_path=model_path, premises=[], batch_size=10)
        proof = prove_htps(handler, state, deepseek_expansion_fn, deepseek_train_fn, num_tactics)
    else:
        premises = vector_store.get_premises(state.goal, K_PREMISES)
        if any(premise.content == conjecture_formalized for premise in premises):
            logger.warning("Premise matched conjecture! %s", conjecture_formalized)
            return False, None
        # Reimport, this time with the premises
        import_strings = [premise.import_string(root_path) for premise in premises]
        full = "\n".join(import_strings) + "\n" + full
        handler = import_code(handler, full)
        # We suffix the theorem since the current name now already exists in the environment because of the lines above
        name_span = theorem_name_span(conjecture_formalized)
        conjecture_formalized = conjecture_formalized[:name_span[1]] + "__" + conjecture_formalized[name_span[1]:]
        logger.info("Executing with premises %s", conjecture_formalized)
        output, env = _send_command(handler, conjecture_formalized)
        assert not "messages" in output or not any(msg.severity == "error" for msg in output["messages"])
        sorries = output["sorries"]
        assert len(sorries) == 1
        state = sorries[0]
        premises = [premise.content for premise in premises]
        reprover_expansion_fn = partial(expand_reprover, premises=premises)
        reprover_train_fn = partial(train_reprover, json_dir=json_path, model_path=model_path, premises=premises)
        proof = prove_htps(handler, state, reprover_expansion_fn, reprover_train_fn, num_tactics)
    # else:

    #     if distributed:
    #         logger.info("Proving with distributed tactic generation")
    #         proven, proof, proofs = prove_distributed(handler, state, [premise.content for premise in premises], comm,
    #                                              num_tactics, proof_path is not None)
    #         proof = proof if proven else None
    #         if proof_path:
    #             # Locking the directory
    #             dir_fd = os.open(proof_path, os.O_RDONLY)
    #             fcntl.flock(dir_fd, fcntl.LOCK_EX)
    #             indices = [int(str(result.stem).split("_")[1]) for result in proof_path.iterdir()]
    #             current_idx = max(indices) + 1 if indices else 0
    #             for idx, result in enumerate(proofs):
    #                 formalized_path = proof_path / f"proof_{idx + current_idx}.json"
    #                 file_fd = os.open(formalized_path, os.O_CREAT | os.O_WRONLY)
    #                 fcntl.flock(file_fd, fcntl.LOCK_EX)
    #                 result.save_json(formalized_path)
    #                 fcntl.flock(file_fd, fcntl.LOCK_UN)
    #             fcntl.flock(dir_fd, fcntl.LOCK_UN)
    #     else:
    #         logger.info("Proving without distributed tactic generation")
    #         proof = prove_reprover(handler, state, [premise.content for premise in premises], num_tactics)
    if proof is not None:
        proven_theorem = ProvenTheorem.from_formalized_conjecture(conjecture, proof)
        if import_strings is not None:
            proven_theorem.imports += import_strings
        return True, proven_theorem
    return False, None


def main():
    from diophantineequations.lemma_embeddings import ReProverEmbeddingFn
    root_path = Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project")
    src_path = Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project/Imo1988q6project")
    store = LemmaVectorStore.from_directory(src_path, ReProverEmbeddingFn())
    # prove_conjecture(root_path,
    #                  FormalizedConjecture("""Assumes: False.
    #                  Shows: False.""", "theorem conjecture_given_false_show_false (h : False) : False := by sorry", []),
    #                  store)
    prove_conjecture(
        root_path,
        FormalizedConjecture("""### Conjecture 3
(Note: Reformulated from your original prompt.)
Given: Set \(X\) and operation \(*\).
Assumes: For all \(x, y \in X\), \((x * y) * x = y\).
Shows: For all \(x, y \in X\), \(x * (y * x) = y\).""", """theorem conjecture_commutative_property (X : Type*) [Mul X] 
(h : ∀ x y : X, (x * y) * x = y) 
(a b : X) : a * (b * a) = b := 
by
 sorry""", []), store)


if __name__ == '__main__':
    main()
