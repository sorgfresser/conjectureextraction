from pydantic import BaseModel, field_validator, model_validator, Field
from enum import IntEnum
from typing import Optional, Dict, Union, List
from lean_interact.interface import Sorry, ProofStepResponse
from diophantineequations.models import ProvenTheorem, LeanFile


class ModelType(IntEnum):
    reprover = 1
    deepseek = 2


class RunConfig(BaseModel):
    retrieval: bool
    dry_run: bool
    search: bool
    recursive: bool


class Conjecture(BaseModel):
    informal_problem: str
    informal_proof: str | None = None
    formal_problem: str | None = None
    formal_proof: str | None = None
    imports: list[str] | None = None
    parent_problem: str | None = None
    attempts: int = 0


class AbstractAction(BaseModel):
    search_idx: int
    run_config: RunConfig


class ActionTactics(AbstractAction):
    theorem: str
    past_tactics: list[str]
    goal: str
    premises: list[str]
    k: int
    model_type: ModelType
    model_path: str

# class ActionTrainSample(AbstractAction):
#     model_type: ModelType
#     model_path: str
#     model_version: int
#     prompt: Union[str, List[Dict[str, str]]] # either string or chat completion
#     generations: List[Union[str], List[Dict[str, str]]]
#     working: list[bool]
#     conjecture: Conjecture
#     context: list[str] | None = None
#     values: list[float] | None = None # exact values for each generation (e.g. logprobs)
#
#     @model_validator(mode="after")
#     def lengths_match(self):
#         if len(self.generations) != len(self.working):
#             raise ValueError("Generations and working must have same length!")
#         if self.values is not None and len(self.values) != len(self.working):
#             raise ValueError("Values and working must have same length!")
#         return self

class ActionTrainSample(AbstractAction):
    model_path: str
    model_type: ModelType
    premises: list[str]
    state: str
    tactic: str

class ActionFormalizationSample(AbstractAction):
    model_path: str
    model_type: ModelType
    formalizations: list[Optional[str]]
    working: list[bool]
    conjecture: Conjecture
    definitions: list[LeanFile]
    prompt: Union[str, List[Dict[str, str]]]


class ActionInitialEnvironment(AbstractAction): # ActionGetProofState
    theorem: str
    past_tactics: list[str]
    context: Optional[str]


class ActionEnvironment(AbstractAction): # ActionProofStep
    theorem: str
    past_tactics: list[str]
    goal: str  # goal is current proof state, i.e. theorem after applying past_tactics
    current_tactics: list[str]
    current_logprobs: list[float]
    context: Optional[str]

class ActionEnvironmentAddHypotheses(AbstractAction): # ActionAddHypotheses
    theorem: str
    decls: list[str]
    context: Optional[str]

class ActionEnvironmentWholeProof(AbstractAction): # ActionWholeProof
    theorem: str
    context: Optional[str]
    proofs: list[str]

class ActionEmbed(AbstractAction):
    theorem: ProvenTheorem
    conjecture: Conjecture


class ActionRetrieve(AbstractAction):
    theorem: str
    past_tactics: list[str]
    goal: str
    k: int
    conjecture: Conjecture

class ActionFormalization(AbstractAction):
    conjecture: Conjecture
    definition_retrieval: bool
    num_formalizations: int

class AbstractActionSearch(AbstractAction):
    theorem: str
    premises: list[str]
    conjecture: Conjecture
    is_inverted: bool = False  # whether or not the current theorem is inverted, for hypothesis rejection


class ActionSearch(AbstractActionSearch):
    num_expansions: int
    num_tactics: int


class ActionNoSearch(AbstractActionSearch):
    max_tactics: int
    num_tactics: int


class ActionConjecture(AbstractAction):
    conjecture: Conjecture
    definition_retrieval: bool


class WorkerAction(BaseModel):
    action: ActionTactics | ActionTrainSample | ActionEnvironment | ActionSearch | ActionInitialEnvironment | ActionConjecture | ActionNoSearch | ActionEmbed | ActionRetrieve | ActionFormalization | ActionFormalizationSample | ActionEnvironmentAddHypotheses | ActionEnvironmentWholeProof


class MasterToWorker(BaseModel):
    message: WorkerAction


class RequestModelUpdate(BaseModel):
    """Let a trainer request that the master updates its underlying model."""
    model_path: str
    model_type: ModelType
    sample_count: int  # total sample count


class WorkerRequest(BaseModel):
    request: RequestModelUpdate


class AbstractResponse(BaseModel):
    action: ActionTactics | ActionEnvironment | ActionSearch | ActionInitialEnvironment
    ms_between: int  # idle time between two actions in milliseconds


class ResponseTactics(AbstractResponse):
    action: ActionTactics
    strings: list[str]
    logprobs: list[float]


class ResponseInitialEnvironment(AbstractResponse):
    action: ActionInitialEnvironment
    proof_state: Optional[Sorry | ProofStepResponse]
    error: Optional[str] = None


class ResponseEnvironment(AbstractResponse):
    action: ActionEnvironment
    next_proof_states: list[Optional[ProofStepResponse]]
    error: Optional[str] = None

class ResponseEnvironmentAddHypotheses(AbstractResponse):
    action: ActionEnvironmentAddHypotheses
    theorem: str
    error: Optional[str] = None

class ResponseEnvironmentWholeProof(AbstractResponse):
    action: ActionEnvironmentWholeProof
    proven: list[bool]
    errors: list[Optional[str]] = Field(default_factory=list)

class ResponseFormalization(AbstractResponse):
    action: ActionFormalization
    attempts: int
    formal_theorem: Optional[str]
    definitions: list[LeanFile]


class AbstractSearchResponse(AbstractResponse):
    action: AbstractActionSearch
    proof: Optional[str]
    error: Optional[str]
    search_time_ms: int


class ResponseNoSearch(AbstractSearchResponse):
    action: ActionNoSearch
    depth: int


class ResponseSearch(AbstractSearchResponse):
    action: ActionSearch
    expansions: int


class ResponseConjecture(AbstractResponse):
    action: ActionConjecture
    conjectures: list[Conjecture]
    attempts: list[int]


class ResponseEmbed(AbstractResponse):
    success: bool
    action: ActionEmbed


class ResponseRetrieve(AbstractResponse):
    action: ActionRetrieve
    files: list[LeanFile]


class WorkerResponse(BaseModel):
    response: ResponseTactics | ResponseEnvironment | ResponseSearch | ResponseInitialEnvironment | ResponseConjecture | ResponseNoSearch | ResponseEmbed | ResponseRetrieve | ResponseFormalization | ResponseEnvironmentAddHypotheses | ResponseEnvironmentWholeProof


class WorkerToMaster(BaseModel):
    message: WorkerResponse | WorkerRequest
