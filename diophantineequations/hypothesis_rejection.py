import weave

from diophantineequations.lemma_prove import prove_conjecture
from diophantineequations.prompts import CONJECTURE_NL_FALSE, CONJECTURE_NL_FALSE_FEW_SHOT
from diophantineequations.utils import conclusion_false
from pathlib import Path
from openai import OpenAI
import anthropic
from diophantineequations.lemma_embeddings import LemmaVectorStore
from diophantineequations.models import FormalizedConjecture
from diophantineequations.distributed_models import WorkerType
import dataclasses

client = OpenAI()


@weave.op()
def natural_language_false(conjecture_nl: str, conjecture: str) -> str:
    messages = [
        {
            "role": "user",
            "content": CONJECTURE_NL_FALSE + "\n\n" + CONJECTURE_NL_FALSE_FEW_SHOT + f"\n\nNatural language conjecture:\n{conjecture_nl}\n\n\nFormal conjecture:\n{conjecture}"
        },
    ]
    # message = client.messages.create(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=1024,
    #     messages=messages
    # )

    # output = "\n".join([block.text for block in message.content])
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    output = result.choices[0].message.content
    return output


@weave.op()
def hypothesis_rejection(root_path: Path, conjecture: FormalizedConjecture, vector_store: LemmaVectorStore,
                         json_path: Path, model_path: Path,
                         distributed: bool = False, comm=None, num_tactics: int = 30, deepseek: bool = False,
                         proof_path: Path | None = None, worker_types: list[WorkerType] | None = None) -> bool:
    falsified = dataclasses.replace(conjecture)
    falsified.formalized_conjecture = conclusion_false(falsified.formalized_conjecture)
    falsified.conjecture = natural_language_false(conjecture.nl_conjecture, falsified.formalized_conjecture)
    proven, _ = prove_conjecture(root_path, falsified, vector_store, json_path, model_path,
                                 distributed=distributed, comm=comm, num_tactics=num_tactics,
                                 proof_path=proof_path, deepseek=deepseek, worker_types=worker_types)
    return proven


def main():
    from diophantineequations.lemma_embeddings import ReProverEmbeddingFn
    conjecture = """theorem conjecture_quadratic_equation_formation (a b k : Nat)
    (h : a > 0 ∧ b > 0 ∧ k > 0)
    (hk : a^2 + b^2 = k * (a * b + 1)) :
     a^2 - k * b * a + b^2 - k = 0 := by
    sorry"""
    nl_conjecture = """Given: Positive integers \( a \), \( b \), \( k \).
Assumes: \( a^2 + b^2 = k(ab + 1) \).
Shows: The equation \( a^2 - k b a + b^2 - k = 0 \) holds."""
    root_path = Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project")
    src_path = Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project/Imo1988q6project")
    store = LemmaVectorStore.from_directory(src_path, ReProverEmbeddingFn())
    conjecture = FormalizedConjecture(nl_conjecture, conjecture, [])
    hypothesis_rejection(root_path, conjecture, store)


if __name__ == '__main__':
    main()
