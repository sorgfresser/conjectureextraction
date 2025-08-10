# import weave
from typing import Tuple, List, Optional, TYPE_CHECKING, Dict
import torch
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from pathlib import Path
import gc
from diophantineequations.environment import DEEPSEEK_DEFAULT_HEADER as LEAN4_DEFAULT_HEADER
from sklearn.metrics import accuracy_score

import time
from diophantineequations.vllmutils import vLLMWrapper
import os
from torch.utils.data import Dataset
from sqlmodel import SQLModel, Field, create_engine, select, Session, Column, JSON, func
from diophantineequations.distributed.messages import ModelType, ActionTrainSample, RunConfig
import logging
import shutil
from diophantineequations.notify import send_notification

if TYPE_CHECKING:
    from transformers import LlamaTokenizerFast, EvalPrediction

logger = logging.getLogger(__name__)


class Sample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_type: ModelType
    model_path: str
    premises: list[str] | None = Field(default=None, sa_column=Column(JSON))
    state: str
    tactic: str
    timestamp: int
    run_config: Dict = Field(default_factory=dict, sa_column=Column(JSON))

    @classmethod
    def from_action(cls, action: ActionTrainSample):
        return cls(model_type=action.model_type, model_path=action.model_path, premises=action.premises,
                   state=action.state, tactic=action.tactic, timestamp=int(time.time()),
                   run_config=action.run_config.model_dump(mode="json"))


class SQLiteDataset(Dataset):
    def __init__(self, db_path: str, run_config: Dict | None = None):
        if not db_path.startswith("sqlite:///"):
            db_path = f"sqlite:///{db_path}"
        self.engine = create_engine(db_path)
        self._run_config = run_config
        with Session(self.engine) as session:
            if run_config is not None:
                stmt = (select(func.count(Sample.id)).where(Sample.run_config == run_config))
            else:
                stmt = select(func.count(Sample.id))
            self._length = session.scalar(stmt)

    def __len__(self):
        return self._length

    def __getitem__(self, item) -> Dict[str, str | List[str]]:
        with Session(self.engine) as session:
            stmt = select(Sample)
            # Optional filter
            if self._run_config is not None:
                stmt = stmt.where(Sample.run_config == self._run_config)
            # Deterministic order, take exactly one row at the desired position
            stmt = stmt.order_by(Sample.id).offset(item).limit(1)
            sample = session.exec(stmt).first()
        return {"state": sample.state, "premises": sample.premises, "tactic": sample.tactic}


def non_cot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )


def non_cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )


def cot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )


def cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )


def post_process_output(output):
    _find_idx = output.find("```")
    return output[:_find_idx] if _find_idx >= 0 else output


MODEL_FORMAT = dict(
    non_cot=dict(prompt=non_cot_prompt, output=post_process_output, few_shot=non_cot_few_shot_prompt),
    cot=dict(prompt=cot_prompt, output=post_process_output, few_shot=cot_few_shot_prompt),
)

mode = MODEL_FORMAT["cot"]
model: Optional[vLLMWrapper] = None
tokenizer: Optional["LlamaTokenizerFast"] = None
device = "cuda"
START_STATEMENT = '<statement>'
START_LEMMA_STMT = '<easy theorem>'
START_THM = '<hard theorem>'
END_THM = '</hard theorem>'
INVOKED_LEMMA = '<lemma>'
PROVER_PROMPT = 'Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'
DEFAULT_MODEL = "kfdong/STP_model_Lean_0320"


def reload_model(model_path: str, vllm_bin: Optional[str] = None):
    from transformers import LlamaTokenizerFast
    global model, tokenizer
    logger.info("Reloading model!")
    model = vLLMWrapper(model_path, "bfloat16", vllm_bin=vllm_bin)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token


def get_model(model_path: Optional[str] = None, vllm_bin: Optional[str] = None):
    global model
    if model_path is None:
        model_path = DEFAULT_MODEL
    if model is None:
        logger.info(f"Loading model from {model_path} in get_model")
        reload_model(model_path, vllm_bin)
    return model


def clean_model():
    global model, tokenizer
    logger.info("Cleaning model!")
    if model is not None:
        logger.info("Model was not none, calling cleanup()!")
        model._cleanup()
    model, tokenizer = None, None
    gc.collect()


def get_prompt(
        statement: str,
        max_length: int,
) -> str:
    prompt = f'{PROVER_PROMPT}\n{statement.strip()}'
    tokens = tokenizer.encode(
        prompt,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    )[0]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# @weave.op()
def generate_tactic(state: str, retrieved_premises: List[str], k: int) -> Tuple[List[str], torch.FloatTensor]:
    """Generates tactics for a given state and retrieved premises.

    :param state: The state to prove.
    :param retrieved_premises: The retrieved premises.
    :param k: The number of tactics to generate.
    """
    logger.debug("Called generate_tactic with state %s", state)
    tokenized_input = get_prompt(state, 1024)
    model_output = model.generate(tokenized_input, k, 0.7, 1024)
    tactics: list[str] = [output.text for output in model_output.choices]
    logger.debug("Generated tactics: %s", tactics)
    # Keep indentation, but remove rest
    tactics = [tactic.strip("\n") for tactic in tactics]
    logprobs_taken = []
    for idx, tactic in enumerate(tactics):
        tactic_probs = torch.tensor(model_output.choices[idx].logprobs.token_logprobs)
        logprobs_taken.append(tactic_probs.sum())
    # Deduplicate
    dedup_tactics = []
    dedup_logprobs = []
    for idx, tactic in enumerate(tactics):
        if tactic in dedup_tactics:
            continue
        dedup_tactics.append(tactic)
        dedup_logprobs.append(logprobs_taken[idx])
    dedup_logprobs = torch.stack(dedup_logprobs)
    logger.debug("Deduplicated tactics: %s", dedup_tactics)
    return dedup_tactics, dedup_logprobs


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tok: "LlamaTokenizerFast", base_dataset: torch.utils.data.Dataset):
        self.tokenizer = tok
        self._ds = base_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, item):
        sample = self._ds[item]
        inputs = "\n".join([PROVER_PROMPT.strip()] + sample["premises"] + [sample["state"].strip()])
        tactic = "\n" + sample["tactic"]
        full = inputs + tactic
        full_tokenized = self.tokenizer(full, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        inputs_tokenized = tokenizer(inputs, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        assert len(full_tokenized.input_ids) == 1, full_tokenized.input_ids.shape
        input_ids = full_tokenized.input_ids[0, :-1]
        attention_mask = full_tokenized.attention_mask[0, :-1]
        labels = full_tokenized.input_ids[0, 1:].clone()
        mask_start = inputs_tokenized.attention_mask.sum() - 1  # -1 because offset by one
        labels[:mask_start] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
    """

    def __init__(self, max_length: int, tokenizer: "LlamaTokenizerFast"):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        assert all(label is not None for label in labels), "Labels not found!"
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Set labels to -100 for padding
        max_len = len(batch.input_ids[0])  # they will all have the correct len
        for idx in range(len(labels)):
            labels[idx] = torch.tensor(labels[idx].tolist() + [-100] * (max_len - len(labels[idx])))
        batch.data["labels"] = torch.stack(labels)
        return batch


def compute_metrics(prediction: "EvalPrediction"):
    predictions = prediction.predictions.argmax(-1)
    total_acc = 0.0
    for idx, pred in enumerate(predictions):
        valid_ids = prediction.label_ids[idx] != -100
        accuracy = accuracy_score(y_true=prediction.label_ids[idx, valid_ids], y_pred=pred[valid_ids])
        total_acc += accuracy
    return {"accuracy": total_acc / len(predictions) if len(predictions) else 0.0}


def train(dataset: torch.utils.data.Dataset, source_model_path: str, result_model_path: str) -> str:
    from transformers import LlamaForCausalLM, LlamaTokenizerFast, TrainingArguments, Trainer
    global model, tokenizer
    train_path = Path(result_model_path) / "training"
    train_path.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(train_path),
        eval_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=2,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        seed=42,
        data_seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["wandb"],
        include_tokens_per_second=False,
        include_num_input_tokens_seen=False,
        bf16=True
    )
    if model:
        clean_model()
    model = LlamaForCausalLM.from_pretrained(source_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = LlamaTokenizerFast.from_pretrained(source_model_path)
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = TokenizedDataset(tokenizer, dataset)
    generator = torch.Generator().manual_seed(42)
    datasets = torch.utils.data.random_split(tokenized_ds, [0.9, 0.1], generator=generator)

    collator = DataCollatorWithPadding(1024, tokenizer)
    trainer = Trainer(
        model=model, args=args, train_dataset=datasets[0], eval_dataset=datasets[1],
        data_collator=collator, compute_metrics=compute_metrics
    )
    trainer.train()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')

    trainer.save_model(result_model_path)
    # model.save_pretrained(result_model_path)
    tokenizer.save_pretrained(result_model_path)
    # Free training
    shutil.rmtree(str(train_path), ignore_errors=True)
    return result_model_path


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("sqlite_path", type=str, help="Path to sqlite database")
    parser.add_argument("source_model_path", type=str, help="Path to source model")
    parser.add_argument("result_model_path", type=str, help="Path to output model")
    parser.add_argument("--run-config", type=str, help="Json run config", default="")
    args = parser.parse_args()
    source_model = args.source_model_path
    result_model = args.result_model_path
    sqlite_path = args.sqlite_path
    run_config_str = args.run_config
    run_config = None
    if run_config_str:
        run_config = RunConfig.model_validate_json(run_config_str)

    import json

    class TacticSamplesDataset(torch.utils.data.Dataset):
        def __init__(self, json_dir: Path):
            self.json_dir = json_dir
            self.files = list(json_dir.glob("*.json"))
            self.file_contents = []
            for f in self.files:
                with f.open() as file:
                    self.file_contents.append(file.read())
            # self._deduplicate()

        def __len__(self):
            return len(self.file_contents)

        def _deduplicate(self):
            self.file_contents = list(dict.fromkeys(self.file_contents))

        def __getitem__(self, item):
            return json.loads(self.file_contents[item])

    # dataset = TacticSamplesDataset(Path("./jsons__deepseek"))
    dataset = SQLiteDataset(sqlite_path, run_config=run_config.model_dump(mode="json"))
    train(dataset, source_model, result_model)


if __name__ == '__main__':
    def is_rank_zero() -> bool:
        """Check if we are distributed and rank 0"""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass  # fallback to env var
        return int(os.environ.get("RANK", "0")) == 0


    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "TrainerInner")
        raise e
    if is_rank_zero():
        send_notification(False, job_name="TrainerInner")
