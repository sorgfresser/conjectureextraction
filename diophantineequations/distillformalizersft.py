from sqlmodel import SQLModel, Field, Relationship, Column, JSON, create_engine, Session, select
from typing import Optional, List, Dict
from sqlalchemy.dialects.postgresql import JSONB
from datasets import Dataset
import re

WS = r"\s+"
THEOREM_REGEX = r"\stheorem[\s(]"


def theorem_name_span(conjecture: str) -> tuple[int, int]:
    """
    Get the span of the theorem name in the given conjecture
    :param conjecture: The conjecture to get the theorem name from, must be without comments
    :return: The span of the theorem name
    """
    data = "\n" + conjecture  # we add this for the upcoming regex, to ensure we get the first theorem
    theorems = list(re.finditer(THEOREM_REGEX, data))
    assert len(theorems) == 1
    start = theorems[0].start()
    data = data[theorems[0].end() - 1:]  # This means the current end is the first ws found in the regex below
    ws = list(re.finditer(WS, data))
    end = ws[1].start() + theorems[0].end() - 2
    return start, end


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


databases = [
    "postgresql+psycopg2://postgres:<postgres-password>@<postgres-ip>:5432/nodeepseekkiminaformalizeputnamonline",
    "postgresql+psycopg2://postgres:<postgres-password>@<postgres-ip>:5432/rabbitnosearchdeepseekkiminaformalize",
    "postgresql+psycopg2://postgres:<postgres-password>@<postgres-ip>:5432/rabbitsearchdeepseekkiminaformalize",
    "postgresql+psycopg2://postgres:<postgres-password>@<postgres-ip>:5432/nodeepseekkiminaformalizeputnam",
]

engines = [create_engine(db) for db in databases]

samples = []
hashes = set()
duplicates = 0

for engine in engines:
    with Session(engine) as session:
        result = session.exec(select(FormalizationSample).where(FormalizationSample.label == True)).all()
        for row in result:
            completions = row.completion
            assert len(completions) == 1, "Expected exactly one completion per sample"
            completion_str = completions[0] if isinstance(completions[0], str) else completions[0]["content"]
            try:
                theorem_name = completion_str[
                               theorem_name_span(completion_str)[0]:theorem_name_span(completion_str)[1]].strip()
            except AssertionError:
                print("Skipping sample with invalid theorem name span:", completion_str)
                continue
            completion_str = "import Mathlib\nimport Aesop\n" + completion_str
            prompt = row.prompt.prompt
            for idx in range(len(prompt)):
                if isinstance(prompt[idx], str):
                    prompt[idx] = prompt[idx].replace("my_favorite_theorem", theorem_name)
                elif isinstance(prompt[idx], dict) and "content" in prompt[idx]:
                    prompt[idx]["content"] = prompt[idx]["content"].replace("my_favorite_theorem", theorem_name)
            if isinstance(completions[0], dict):
                completions[0]["content"] = completion_str
            else:
                completions[0] = completion_str
            # Hash prompt and completion to avoid duplicates
            prompt_str = "\n".join(prompt) if isinstance(prompt[0], str) else "\n".join([p["content"] for p in prompt])
            completion_str = completions[0] if isinstance(completions[0], str) else completions[0]["content"]
            prompt_hash = hash(prompt_str)
            completion_hash = hash(completion_str)
            if (prompt_hash, completion_hash) in hashes:
                duplicates += 1
                continue
            hashes.add((prompt_hash, completion_hash))
            samples.append({"prompt": row.prompt.prompt, "completion": completions})
print(f"Found {len(samples)} samples across all databases.")
print(f"Found {duplicates} duplicates across all databases.")

dataset = Dataset.from_list(samples)
dataset.push_to_hub("sorgfresser/formalizationsamples")
