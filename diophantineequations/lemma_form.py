from openai import OpenAI
from diophantineequations.prompts import FORMALIZATION_CONTEXT, FORMALIZATION_FEW_SHOT, CONJECTURE_FORMALIZATION
from diophantineequations.environment import get_handler, import_code
from pathlib import Path
from logging import getLogger
from typing import List, Union, Optional
from diophantineequations.definition_retrieval import LeanFile
from diophantineequations.models import FormalizedConjecture
import fcntl
import os
import weave

logger = getLogger(__name__)

client = OpenAI()
MAX_TRIES = 20
AT_A_TIME = 5

def generate_formalizations(conjecture: str, context: Union[List[LeanFile], None], tries: int = AT_A_TIME) -> List[Optional[str]]:
    if context:
        context_prompt = "\n\n".join(["```lean4" + c.content + "```" for c in context])
        messages = [
            {
                "role": "user",
                "content": FORMALIZATION_CONTEXT + "\n" + CONJECTURE_FORMALIZATION + "\n" + FORMALIZATION_FEW_SHOT + "\n\n" + "Here are some definitions you might find helpful:\n\n" + context_prompt + "\n\n" + conjecture
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": FORMALIZATION_CONTEXT + "\n" + CONJECTURE_FORMALIZATION + "\n" + FORMALIZATION_FEW_SHOT + "\n\n" + conjecture
            }
        ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=messages,
        n=tries,
        seed=42
    )
    choices = response.choices
    results = []
    for choice in choices:
        output = choice.message.content
        try:
            lean_block = output.split("```lean")[1].split("```")[0]
            # Means ```lean4 instead of ```lean
            if lean_block.startswith("4"):
                lean_block = lean_block.removeprefix("4")
            results.append(lean_block)
        except IndexError:
            logger.exception("IndexError when trying to parse LLM for Formalization output")
            logger.info("Output that raised IndexError: %s", output)
            results.append(None)
    return results


weave.op()
def formalize_conjecture(root_path: Path, conjecture: str,
                         context: Union[List[LeanFile], None] = None,
                         result_dir_path: Union[Path, None] = None) -> FormalizedConjecture | None:
    """Uses 4o-mini to formalize a natural language conjecture

    :param root_path: Path to the root directory of the project
    :param conjecture: Conjecture in natural language
    :param context: List of Lean files to use as context for the formalization, or None
    :param result_dir_path: Path to the directory where the formalized conjecture should be saved or None
    :return: Formalized conjecture, not checked for syntax etc.
    """
    logger.info("Formalizing conjecture %s", conjecture)
    handler = get_handler(root_path)
    imports: List[str] = [file.import_string(root_path).strip() for file in context] if context else []
    if context:
        handler = import_code(handler, "\n".join(imports))

    for _ in range(0, MAX_TRIES, AT_A_TIME):
        lean_blocks = generate_formalizations(conjecture, context)
        for lean_block in lean_blocks:
            if lean_block is None:
                continue
            handler.send_command(lean_block)
            repl_result, env = handler.receive_json()
            # We expect one message stating that we're using a sorry, and maybe some additional messages
            messages = repl_result.get("messages", [])
            has_sorry_warning = any("declaration uses 'sorry'" in message.data for message in messages)
            # If any error is found, we need to retry
            has_no_errors = all(message.severity != "error" for message in messages)
            sorries = repl_result.get("sorries", [])
            has_exactly_one_sorry = len(sorries) == 1
            logger.debug("Formalization %s, has_sorry_warning: %s, has_no_errors: %s, has_exactly_one_sorry: %s",
                         lean_block, has_sorry_warning, has_no_errors, has_exactly_one_sorry)
            if has_sorry_warning and has_no_errors and has_exactly_one_sorry:
                logger.info("Formalization %s successful", lean_block)
                result = FormalizedConjecture(conjecture, lean_block, imports)
                if result_dir_path:
                    # Locking the directory
                    dir_fd = os.open(result_dir_path, os.O_RDONLY)
                    fcntl.flock(dir_fd, fcntl.LOCK_EX)
                    indices = [int(str(result.stem).split("_")[1]) for result in result_dir_path.iterdir()]
                    idx = max(indices) + 1 if indices else 0
                    formalized_path = result_dir_path / f"formalized_{idx}.json"
                    file_fd = os.open(formalized_path, os.O_CREAT | os.O_WRONLY)
                    fcntl.flock(file_fd, fcntl.LOCK_EX)
                    result.save_json(formalized_path)
                    fcntl.flock(file_fd, fcntl.LOCK_UN)
                    fcntl.flock(dir_fd, fcntl.LOCK_UN)
                return result

            logger.warning("Formalization failed, retrying!")
            logger.debug("Formalization %s failed with repl result %s", lean_block, repl_result)

    return None


def main():
    conjecture = """
**Conjecture 2: Minimality Implies Bound on the Other Root**

- **Given:** Positive integers \( a \), \( b \), \( k \), with \( a \geq b \), and integer \( c \).
- **Assumes:**
  - \( a^2 - k b a + b^2 - k = 0 \).
  - \( c = k b - a \) (the other root of the quadratic equation).
- **Shows:** \( c \leq 0 \).
"""
    root_path = Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project")
    print(formalize_conjecture(root_path, conjecture,
                               result_dir_path=Path("/home/simon/PycharmProjects/diophantineequations/formalized")))



if __name__ == '__main__':
    main()
    # conjecture = """**Conjecture 2: Minimality Implies Bound on the Other Root**
    #
    #     - **Given:** Positive integers \( a \), \( b \), \( k \), with \( a \geq b \), and integer \( c \).
    #     - **Assumes:**
    #       - \( a^2 - k b a + b^2 - k = 0 \).
    #       - \( c = k b - a \) (the other root of the quadratic equation).
    #     - **Shows:** \( c \leq 0 \)."""
    # formalize_conjecture(conjecture)
#     conjecture = """**Conjecture 1: Quadratic Equation Formation**
#
# - **Given:** Positive integers \( a \), \( b \), \( k \).
# - **Assumes:** \( a^2 + b^2 = k(ab + 1) \).
# - **Shows:** The equation \( a^2 - k b a + b^2 - k = 0 \) holds.
# """
#     print(formalize_conjecture(conjecture))
