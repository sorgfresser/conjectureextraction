from typing import Union, List, Optional
from diophantineequations.models import LeanFile
from diophantineequations.vllmutils import vLLMWrapper
from diophantineequations.distributed.workers.formalizer.abstract import AbstractFormalizationWorker, Conjecture
from diophantineequations.distributed.messages import MasterToWorker, WorkerAction, ActionFormalizationSample, ModelType
from diophantineequations.notify import send_notification
import torch
import logging
from pathlib import Path
from os import getenv
from uuid import uuid4
import gc

MAX_TRIES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model: Optional[vLLMWrapper] = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reload_model(model_name: str, vllm_bin: str | None = None):
    global model
    logger.info("Reloading model %s", model_name)
    model = vLLMWrapper(model_name, "bfloat16", vllm_bin)
    return model


def get_model(model_name: str, vllm_bin: str | None = None):
    global model
    if model is None:
        reload_model(model_name, vllm_bin)
    return model

def get_prompt_messages(conjecture: str) -> list[dict[str, str]]:
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt += conjecture
    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": prompt}
    ]
    return messages

def generate_formalizations(conjecture: str, context: Union[List[LeanFile], None], tries: int = MAX_TRIES) -> List[
    Optional[str]]:
    messages = get_prompt_messages(conjecture)
    model_output = model.chat_completion(messages, tries, 0.7, 4096)
    uuid_str = str(uuid4()).split("-")[0]
    formalizations: list[str] = [output.message.content.replace("my_favorite_theorem", f"theorem_{uuid_str}")
                                 for output in model_output.choices]
    logger.debug("Generated formalizations: %s", formalizations)
    return formalizations


def clean_model():
    global model, tokenizer
    logger.info("Cleaning model!")
    if model is not None:
        logger.info("Model was not none, calling cleanup()!")
        model._cleanup()
    model, tokenizer = None, None
    gc.collect()


class FormalizerKimina(AbstractFormalizationWorker):
    def __init__(self, trainer_exchange: str, initial_model_path: str, db_path: str, definition_path: Path,
                 vllm_bin: str, max_attempts: int = 10):
        super().__init__(trainer_exchange, initial_model_path, db_path, definition_path, max_attempts)
        self._vllm_bin = vllm_bin

    def get_formalizations(self, conjecture: Conjecture, definitions: list[LeanFile]) -> list[Optional[str]]:
        get_model(self.model_path, self._vllm_bin)
        formalized = generate_formalizations(conjecture.informal_problem, definitions, tries=self._max_attempts)
        formalized_lines = [
            [line.strip() for line in formalproblem.split("\n") if not line.startswith("import") and line.strip()]
            if formalproblem is not None else None for formalproblem in formalized]
        formalized = ["\n".join(formalproblem_lines) if formalproblem_lines else None for formalproblem_lines in
                      formalized_lines]
        return formalized

    def send_response(self, formalization: Optional[str], attempt: int, conjecture: Conjecture):
        prompt = get_prompt_messages(conjecture.informal_problem)
        msg = MasterToWorker(message=WorkerAction(action=ActionFormalizationSample(model_path=self.model_path,
                                                                                   search_idx=self._action.search_idx,
                                                                                   run_config=self._action.run_config,
                                                                                   formalizations=self._formalizations,
                                                                                   definitions=self._definitions,
                                                                                   conjecture=self._action.conjecture,
                                                                                   prompt=prompt,
                                                                                   working=self._working,
                                                                                   model_type=ModelType.deepseek)))
        self._send_to_trainer(msg)
        super().send_response(formalization, attempt, conjecture)

    @classmethod
    def from_env_vars(cls):
        definition_path = getenv("DEFINITION_PATH")
        if definition_path is None:
            raise ValueError("Definition path not set!")
        definition_path = Path(definition_path)
        if not definition_path.exists():
            raise ValueError("Definition path not found!")
        vllm_bin = getenv("VLLM_BIN")
        if vllm_bin is None:
            raise ValueError("VLLM_BIN not set!")
        trainer_exchange = getenv("RABBIT_FORMALIZATION_TRAINER")
        if trainer_exchange is None:
            raise ValueError("Trainer exchange not set!")
        db_url = getenv("POSTGRES_URL")
        if db_url is None:
            raise ValueError("Database url not set!")
        initial_model_path = getenv("MODEL_PATH")
        if initial_model_path is None:
            raise ValueError("Initial model path not provided!")

        return cls(trainer_exchange, initial_model_path, db_url, definition_path, vllm_bin)


def main():
    worker = FormalizerKimina.from_env_vars()
    worker.start()


if __name__ == '__main__':
    from diophantineequations.distributed.gpu_lock import reserve_one_gpu
    import traceback

    try:
        reserve_one_gpu()
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "KiminaFormalizer")
        raise e
    send_notification(False, job_name="KiminaFormalizer")
