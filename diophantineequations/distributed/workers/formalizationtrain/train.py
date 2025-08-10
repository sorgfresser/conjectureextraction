import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollator
from diophantineequations.distributed.messages import ActionFormalizationSample, Conjecture, RunConfig
from diophantineequations.distributed.workers.formalizationtrain.grposinglestep import GRPOTrainer, GRPOConfig
import deepspeed
from typing import List, Dict, Optional, Tuple, Any
from typing_extensions import Self
import torch
from sqlmodel import SQLModel, Field, Column, JSON, create_engine, Session, select, func, Relationship
from sqlalchemy.dialects.postgresql import JSONB
import time
import os
from diophantineequations.notify import send_notification
from pathlib import Path
import shutil
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from types import MethodType
import logging
from trl.data_utils import maybe_apply_chat_template
from diophantineequations.distributed.workers.formalizationtrain.similarity import get_similarities, load_model, format_doc_only_f

logger = logging.getLogger(__name__)


class Prompt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str | List[Dict[str, str]] = Field(default_factory=str, sa_column=Column(JSON))
    run_config: Dict = Field(default_factory=dict, sa_type=JSONB)
    version: int
    samples: List["FormalizationSample"] = Relationship(back_populates="prompt")


class FormalizationSample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    completion: str | List[Dict[str, str]] = Field(default_factory=str, sa_column=Column(JSON))
    label: bool
    timestamp: int
    prompt: Prompt = Relationship(back_populates="samples")
    prompt_id: Optional[int] = Field(default=None, foreign_key="prompt.id")

    @classmethod
    def from_action(cls, action: ActionFormalizationSample, version_idx: int) -> Tuple[Prompt, List[Self]]:
        result = []
        prompt = Prompt(prompt=action.prompt, run_config=action.run_config.model_dump(mode="json"), version=version_idx)
        for idx, form in enumerate(action.formalizations):
            if form is None:
                continue
            if isinstance(action.prompt, list):
                completion = [{"role": "assistant", "content": form}]
            else:
                completion = form
            result.append(
                cls(prompt=prompt, completion=completion, label=action.working[idx], timestamp=int(time.time())))
        return prompt, result


class SQLDataset(torch.utils.data.Dataset):
    def __init__(self, db_url: str, tokenizer, run_config: Dict | None = None, version: int | None = None,
                 scale_rewards: bool = True):
        self.db_url = db_url
        self._run_config = run_config
        self._version = version
        self.tokenizer = tokenizer
        self.engine = None
        engine = create_engine(db_url)
        with Session(engine) as session:
            stmt = select(func.count(FormalizationSample.id)).join(Prompt)
            if run_config is not None:
                stmt = stmt.where(Prompt.run_config == run_config)
            if version is not None:
                stmt = stmt.where(Prompt.version == version)
            self._length = session.scalar(stmt)
        self._scale_rewards = scale_rewards
        engine.dispose()

    def __len__(self):
        return self._length

    def _scale(self, prompt: Prompt, sample: FormalizationSample) -> float:
        samples = prompt.samples
        rewards = torch.tensor([1 if sample.label else 0 for sample in samples], dtype=torch.float32)
        # Map means and stds back to original reward positions
        sample_reward = 1 if sample.label else 0
        advantage = sample_reward - rewards.mean()
        if self._scale_rewards:
            advantage /= rewards.std() + 1e-4
        return advantage.item()

    def __getitem__(self, item) -> Dict[str, str | List[str] | float]:
        if self.engine is None:
            self.engine = create_engine(self.db_url)
        with Session(self.engine) as session:
            stmt = select(FormalizationSample, Prompt).join(Prompt)
            # Optional filter
            if self._run_config is not None:
                stmt = stmt.where(Prompt.run_config == self._run_config)
            if self._version is not None:
                stmt = stmt.where(Prompt.version == self._version)
            # Deterministic order, take exactly one row at the desired position
            stmt = stmt.order_by(FormalizationSample.id).offset(item).limit(1)
            sample, prompt = session.exec(stmt).first()
            advantage = self._scale(prompt, sample)
        return {"prompt": prompt.prompt, "completion": sample.completion, "reward": advantage}

    def mean_reward(self) -> float:
        engine = create_engine(self.db_url)
        with Session(engine) as session:
            stmt = select(func.count(FormalizationSample.label)).join(Prompt)
            # Optional filter
            if self._run_config is not None:
                stmt = stmt.where(Prompt.run_config == self._run_config)
            if self._version is not None:
                stmt = stmt.where(Prompt.version == self._version)
            stmt = stmt.where(FormalizationSample.label == True)
            # Deterministic order, take exactly one row at the desired position
            num = session.scalar(stmt)
        engine.dispose()
        return float(num) / len(self)

    def similarity(self) -> dict[str, float]:
        engine = create_engine(self.db_url)
        with Session(engine) as session:
            stmt = select(FormalizationSample).join(Prompt)
            if self._run_config is not None:
                stmt = stmt.where(Prompt.run_config == self._run_config)
            if self._version is not None:
                stmt = stmt.where(Prompt.version == self._version)
            samples = session.exec(stmt).all()
            inputs = [{"prompt": sample.prompt.prompt, "completion": sample.completion} for sample in samples]
        engine.dispose()
        informal_statements = [maybe_apply_chat_template(sample, self.tokenizer)["prompt"] for sample in inputs]
        formal_statements = [format_doc_only_f(maybe_apply_chat_template(sample, self.tokenizer)["completion"]) for sample in inputs]
        load_model()
        sims = get_similarities([formal_statements], informal_statements)[0]
        avg_sim = sims.mean().item()  # is averaging over all completions per prompt
        max_sim = sims.max(dim=1).values.mean().item()  # averaging over best completion per prompt
        min_sim = sims.min(dim=1).values.mean().item()  # min completion per prompt
        return {"avg_completion": avg_sim, "max_completion": max_sim, "min_completion": min_sim}


LOCAL_RANK: int = -1


def is_rank_zero() -> bool:
    """Check if we are distributed and rank 0"""
    if LOCAL_RANK == 0:
        return True
    if LOCAL_RANK > 0:
        return False
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass  # fallback to env var
    return int(os.environ.get("RANK", "0")) == 0


# class UnpairedPreferenceDataset(torch.utils.data.Dataset):
#     def __init__(self, samples: list[ActionFormalizationSample]):
#         self.preferences = [get_unpaired_preferences(sample) for sample in samples]
#
#     def __len__(self):
#         return len(self.preferences)
#
#     def __getitem__(self, idx):
#         return self.preferences[idx]

#
# @dataclass
# class DataCollatorPreferences:
#     def __call__(self, features: List[List[UnpairedPreference]]):
#         def to_dict(samples: List[UnpairedPreference]):
#             return [asdict(sample) for sample in samples]
#
#         return sum(map(to_dict, features), [])


#
# def _get_per_token_logps(temperature, model, input_ids, attention_mask, logits_to_keep):
#     # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
#     logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
#     logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
#
#     input_ids = input_ids[:, -logits_to_keep:]
#     # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
#     # See https://github.com/huggingface/trl/issues/2770
#     logits = logits[:, -logits_to_keep:]
#     # Divide logits by sampling temperature.
#     # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
#     logits = logits / temperature
#     return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
#
#
# def compute_loss(beta, model, inputs, temperature, epsilon_low, epsilon_high,
#                  return_outputs=False, num_items_in_batch=None, num_iterations=1):
#     if return_outputs:
#         raise ValueError("The GRPOTrainer does not support returning outputs")
#     # Compute the per-token log probabilities for the model
#     prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
#     completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
#     input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
#     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
#     logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
#
#     per_token_logps = _get_per_token_logps(temperature, model, input_ids, attention_mask, logits_to_keep)
#
#     # Compute the KL divergence between the model and the reference model
#     if beta != 0.0:
#         ref_per_token_logps = inputs["ref_per_token_logps"]
#         per_token_kl = (
#                 torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
#         )
#
#     # Compute the loss
#     advantages = inputs["advantages"]
#     # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
#     # _generate_and_score_completions) and use per_token_logps.detach() instead.
#     old_per_token_logps = inputs["old_per_token_logps"] if num_iterations > 1 else per_token_logps.detach()
#     coef_1 = torch.exp(per_token_logps - old_per_token_logps)
#     coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
#     per_token_loss1 = coef_1 * advantages.unsqueeze(1)
#     per_token_loss2 = coef_2 * advantages.unsqueeze(1)
#     per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
#     if beta != 0.0:
#         per_token_loss = per_token_loss + beta * per_token_kl
#     loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
#
#     # Log the metrics
#     mode = "eval" if self.control.should_evaluate else "train"
#
#     if beta != 0.0:
#         mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
#         self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
#
#     is_clipped = (per_token_loss1 < per_token_loss2).float()
#     clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
#     self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
#     return loss

def is_compiled_module(module):
    """
    Check whether the module was compiled with torch.compile()
    """
    if not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def extract_model_from_parallel(
        model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = True, recursive: bool = False
):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.
        keep_torch_compile (`bool`, *optional*):
            Whether to unwrap compiled model.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel, deepspeed.DeepSpeedEngine, FSDP)

    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod

    while isinstance(model, options):
        model = model.module

    if recursive:
        # This is needed in cases such as using FSDPv2 on XLA
        def _recursive_unwrap(module):
            # Wrapped modules are standardly wrapped as `module`, similar to the cases earlier
            # with DDP, DataParallel, DeepSpeed, and FSDP
            if hasattr(module, "module"):
                unwrapped_module = _recursive_unwrap(module.module)
            else:
                unwrapped_module = module
            # Next unwrap child sublayers recursively
            for name, child in unwrapped_module.named_children():
                setattr(unwrapped_module, name, _recursive_unwrap(child))
            return unwrapped_module

        # Start with top-level
        model = _recursive_unwrap(model)

    if not keep_fp32_wrapper:
        forward = model.forward
        original_forward = model.__dict__.pop("_original_forward", None)
        if original_forward is not None:
            while hasattr(forward, "__wrapped__"):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = MethodType(forward, model)

    if keep_torch_compile and is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model

    return model


def get_state_dict(model, config, unwrap=True):
    """
    Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
    precision.

    Args:
        model (`torch.nn.Module`):
            A PyTorch model sent through [`Accelerator.prepare`]
        unwrap (`bool`, *optional*, defaults to `True`):
            Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict

    Returns:
        `dict`: The state dictionary of the model potentially without full precision.

    Example:

    ```python
    >>> import torch
    >>> from accelerate import Accelerator

    >>> accelerator = Accelerator()
    >>> net = torch.nn.Linear(2, 2)
    >>> net = accelerator.prepare(net)
    >>> state_dict = accelerator.get_state_dict(net)
    ```
    """

    zero3_sharding = config["zero_optimization"]["stage"] == 3
    tp_sharding = config.get("tensor_parallel", {}).get("autotp_size", 0) > 1
    if zero3_sharding or tp_sharding:
        if model.zero_gather_16bit_weights_on_model_save():
            state_dict = (
                model._consolidated_16bit_state_dict()
                if tp_sharding
                else model._zero3_consolidated_16bit_state_dict()
            )
        else:
            raise ValueError(
                "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
            )
    else:
        from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
        state_dict = clone_tensors_for_torch_save(extract_model_from_parallel(model, True, True).state_dict())
    return state_dict


def train(args, dataset: SQLDataset, source_model: str, result_model: str):
    # def collator(features: List[List[Dict[str, Any]]]):
    #     return sum(features, [])

    train_path = Path(result_model) / "training"
    train_path.mkdir(parents=True, exist_ok=True)
    # data_collator = DataCollatorPreferences()
    training_args = GRPOConfig(output_dir=str(train_path), logging_steps=10, num_train_epochs=1, bf16=True)
    tokenizer = AutoTokenizer.from_pretrained(source_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # trainer = GRPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset,
    #                       reward_funcs=[],
    #                       data_collator=data_collator)
    trainer = GRPOTrainer(model_id=source_model, args=training_args, processing_class=tokenizer, train_dataset=dataset,
                          config=args.deepspeed_config)
    trainer.train()
    config = deepspeed.DeepSpeedConfig(args.deepspeed_config)._param_dict
    state_dict = get_state_dict(trainer.model, config)
    if is_rank_zero():
        extract_model_from_parallel(trainer.model).save_pretrained(
            result_model, safe_serialization=training_args.save_safetensors, state_dict=state_dict
        )
        tokenizer.save_pretrained(result_model)
    # Free training
    shutil.rmtree(str(train_path), ignore_errors=True)
    return result_model


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("sql_path", type=str, help="sql database connection string")
    parser.add_argument("source_model_path", type=str, help="Path to source model")
    parser.add_argument("result_model_path", type=str, help="Path to output model")
    parser.add_argument("--run-config", type=str, help="Json run config", default="")
    parser.add_argument("--version", type=int, help="Version index", default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    global LOCAL_RANK
    LOCAL_RANK = args.local_rank
    source_model = args.source_model_path
    result_model = args.result_model_path
    sql_path = args.sql_path
    run_config_str = args.run_config
    run_config = None
    if run_config_str:
        run_config = RunConfig.model_validate_json(run_config_str)
    version = args.version
    tokenizer = AutoTokenizer.from_pretrained(source_model)
    dataset = SQLDataset(sql_path, tokenizer,
                         run_config=run_config.model_dump(mode="json") if run_config else None,
                         version=version if version > - 1 else None)
    if is_rank_zero():
        logger.warning("Starting training for %s samples", len(dataset))
        wandb.init(project="diophantine")
        wandb.log({"mean_reward": dataset.mean_reward(), **dataset.similarity()})
    train(args, dataset, source_model, result_model)


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "TrainerInner")
        raise e
    if is_rank_zero():
        send_notification(False, job_name="TrainerInner")

# deepspeed --num-nodes 1 --num-gpus 4 train.py samples.sqlite3 AI-MO/Kimina-Autoformalizer-7B models --deepspeed_config deepspeedconf.json
