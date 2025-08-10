import gc

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTextEncoding, T5ForConditionalGeneration, ByT5Tokenizer, T5EncoderModel
import torch
from typing import List, Optional, Tuple
import  logging
import re
import weave
from pathlib import Path

logger = logging.getLogger(__name__)
tokenizer_tacgen: Optional[ByT5Tokenizer] = None
model_tacgen: Optional[T5ForConditionalGeneration] = None

tokenizer_retriever: Optional[ByT5Tokenizer] = None
model_retriever: Optional[T5EncoderModel] = None

PREMISE_SUFFIX  = r"(\w+)(\</a\>?)"

@weave.op()
def _remove_premise_suffix(tactic: str) -> str:
    # keep group 0, remove the </a> part
    return re.sub(PREMISE_SUFFIX, r"\1", tactic)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clean_model():
    global tokenizer_tacgen, tokenizer_retriever, model_tacgen, model_retriever
    tokenizer_tacgen, tokenizer_retriever, model_tacgen, model_retriever = None, None, None, None
    gc.collect()

def reload_model(model_path: str):
    global tokenizer_tacgen, model_tacgen
    logger.info("Loading ReProver tacgen tokenizer from %s", model_path)
    tokenizer_tacgen = AutoTokenizer.from_pretrained(model_path)
    logger.info("Loading ReProver tacgen model from %s", model_path)
    model_tacgen = T5ForConditionalGeneration.from_pretrained(model_path)
    model_tacgen.to(DEVICE)

def get_model(model_path: Optional[str] = None):
    if model_path is None:
        model_path = "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
    global tokenizer_tacgen, tokenizer_retriever, model_tacgen, model_retriever
    if tokenizer_tacgen is None or model_tacgen is None:
        reload_model(model_path)
    if tokenizer_retriever is None:
        logger.info("Loading ReProver retriever tokenizer")
        tokenizer_retriever = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
    if model_retriever is None:
        logger.info("Loading ReProver retriever model")
        model_retriever = AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
        model_retriever.to(DEVICE)


@torch.no_grad()
def embed(premises: List[str]):
    tokenized = tokenizer_retriever(premises, return_tensors="pt", padding=True).to(DEVICE)
    output = model_retriever(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
    hidden_states = output.last_hidden_state
    # Single feature
    seq_lens = tokenized.attention_mask.sum(dim=1)
    features = (hidden_states * tokenized.attention_mask.unsqueeze(2)).sum(dim=1) / seq_lens.unsqueeze(1)
    return features


@torch.no_grad()
def retrieve(state: str, premises: List[str], k: int) -> List[str]:
    state_emb = embed([state])
    premise_embs = embed(premises)
    scores = (state_emb.squeeze() @ premise_embs.T)
    topk = scores.topk(k).indices.tolist()
    return [premises[i] for i in topk]


@weave.op()
def generate_tactic(state: str, retrieved_premises: List[str], k: int) -> Tuple[List[str], torch.FloatTensor]:
    """Generates tactics for a given state and retrieved premises.

    :param state: The state to prove.
    :param retrieved_premises: The retrieved premises.
    :param k: The number of tactics to generate.
    """
    input_str = "\n\n".join(retrieved_premises + [state])
    tokenized_input = tokenizer_tacgen(input_str, return_tensors="pt", max_length=2300, truncation=True).to(DEVICE)
    length_penalty = 0.0 if k > 1 else None
    model_output = model_tacgen.generate(tokenized_input.input_ids, max_length=1024,
                                       num_beams=k, length_penalty=length_penalty, do_sample=False, num_return_sequences=k,
                                       early_stopping=False, return_dict_in_generate=True, output_scores=True)
    tactic_ids = model_output.sequences
    tactic_candidates = tokenizer_tacgen.batch_decode(
        tactic_ids, skip_special_tokens=True
    )
    tactics = [_remove_premise_suffix(tactic) for tactic in tactic_candidates]
    return tactics, model_output.sequences_scores


def train(data_loader: torch.utils.data.DataLoader, source_model_path: str, result_model_path: str) -> str:
    global model_tacgen, tokenizer_tacgen
    model_tacgen = T5ForConditionalGeneration.from_pretrained(source_model_path)
    tokenizer_tacgen = AutoTokenizer.from_pretrained(source_model_path)
    model_tacgen.to(DEVICE)
    optimizer = torch.optim.AdamW(model_tacgen.parameters(), lr=1e-5)
    for batch in data_loader:
        inputs = ["\n\n".join(sample["premises"] + [sample["state"]]) for sample in batch]
        tactics = [sample["tactic"] for sample in batch]
        inputs_tokenized = tokenizer_tacgen(inputs, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(DEVICE)
        tactics_tokenized = tokenizer_tacgen(tactics, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(DEVICE)
        tactic_ids = tactics_tokenized.input_ids
        for i in range(len(tactics)):
            tactic_ids[i, ~tactics_tokenized.attention_mask[i]] = -100
        loss = model_tacgen(input_ids=inputs_tokenized.input_ids, attention_mask=inputs_tokenized.attention_mask, labels=tactic_ids).loss
        loss.backward()
        optimizer.step()
    model_tacgen.save_pretrained(result_model_path)
    tokenizer_tacgen.save_pretrained(result_model_path)
    return result_model_path
