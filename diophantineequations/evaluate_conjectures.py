import anthropic
from diophantineequations.prompts import CONJECTURE_EVALUATION
from logging import getLogger
import re

logger = getLogger(__name__)
client = anthropic.Client()

relevance_regex = r"Relevance: \d?\d"


def evaluate_conjecture(conjecture_formal: str, conjecture_nl: str, should_be_used_in: str) -> int:
    logger.info("Evaluating formal conjecture %s", conjecture_formal)
    logger.debug("Evaluating conjecture. Formal: %s\nNL: %s\nShould be used in: %s", conjecture_formal, conjecture_nl,
                 should_be_used_in)
    messages = [{
        "role": "user",
        "content": CONJECTURE_EVALUATION + "\n\nFormal conjecture\n" + conjecture_formal + "\n\nNatural language conjecture\n" + conjecture_nl + "\n\nFinal theorem\n" + should_be_used_in
    }]
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages
    )
    output = "\n".join([block.text for block in message.content])
    logger.info("Received evaluation from LLM.")
    logger.debug("Evaluation from LLM: %s", output)
    match = re.findall(relevance_regex, output)
    logger.debug("Extracted relevance matches: %s", match)
    if len(match) != 1:
        logger.warning("Not exactly one match for the relevance regex on output\n %s", output)
    matched_string: str = match[0]
    relevance_score = matched_string.removeprefix("Relevance: ").strip()
    logger.debug("Relevance score: %s", relevance_score)
    return int(relevance_score)


def main():
    print(evaluate_conjecture("""theorem conjecture_given_false_show_false (h : False) : False := by
sorry""", """Assumes: False.
Shows: False.""",
                              """-- If `ab + 1` divides `a^2 + b^2`, then the quotient `q = (a^2 + b^2) / (ab + 1)` is a perfect square.
                              theorem perfect_square_of_divides (a b : Nat) (h : a > 0 ∧ b > 0) (hdiv : ab + 1 ∣ a^2 + b^2) :
                                  ∃ k : Nat, (a^2 + b^2) / (ab + 1) = k^2 := by
                                  sorry"""))


if __name__ == '__main__':
    main()
