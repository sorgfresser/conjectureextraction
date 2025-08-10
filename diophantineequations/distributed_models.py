from pydantic import BaseModel
from pathlib import Path
from diophantineequations.models import FormalizedConjecture, ProvenTheorem
from enum import Enum

class ModelType(Enum):
    reprover = 1
    deepseek = 2

class WorkerType(Enum):
    environment = 1
    trainer = 2


class WorkerRequestInit(BaseModel):
    worker_type: WorkerType


class WorkerRequest(BaseModel):
    request: WorkerRequestInit


class WorkerReady(BaseModel):
    pass


class WorkerAction(BaseModel):
    pass


class ActionInit(WorkerAction):
    problem_id: str
    model_type: ModelType


class ActionStop(WorkerAction):
    pass


class ActionProve(WorkerAction):
    root_path: Path
    conjecture: FormalizedConjecture


class ActionTactics(WorkerAction):
    goal: str
    premises: list[str]
    k: int

class ActionReload(WorkerAction):
    model_type: ModelType
    model_path: Path


class WorkerMessage(BaseModel):
    action: ActionInit | ActionStop | ActionProve | ActionTactics | ActionReload


class ProofResponse(BaseModel):
    proven: bool
    proof: ProvenTheorem | None


class TacticsResponse(BaseModel):
    tactics: list[str]
    tactic_scores: list[float]
    goal: str


class WorkerResponse(BaseModel):
    response: ProofResponse | TacticsResponse | WorkerReady


class ActionTrain(BaseModel):
    model_path: Path
    model_type: ModelType
    dataset_path: Path

class TrainerMessage(BaseModel):
    action: ActionInit | ActionStop | ActionTrain


class TrainerRequest(BaseModel):
    request: WorkerRequestInit

class TrainedResponse(BaseModel):
    model_path: Path
    model_type: ModelType
    dataset_path: Path

class TrainerResponse(BaseModel):
    response: TrainedResponse
