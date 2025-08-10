from openai import OpenAI
from diophantineequations.prompts import GEN_CONTEXT, CONJECTURE_GEN
from logging import getLogger
import weave
client = OpenAI()

logger = getLogger(__name__)

@weave.op()
def generate_conjectures(problem:str, proof:str, model: str = "o1-preview") -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": GEN_CONTEXT + "\n" + CONJECTURE_GEN + "\n\n" + f"Problem statement\n{problem}" + "\n\n" + f"Proof statement:\n{proof}"
            }
        ],
        seed=42
    )
    result = response.choices[0].message.content
    logger.info("Received conjectures from LLM.")
    logger.debug("Conjectures from LLM: %s", result)
    reasonings = result.split("**Reasoning:**")
    reasonings.pop(0)
    reasonings = ["**Reasoning:**" + r for r in reasonings]
    conjectures_reasoning = [r.split("### Conjecture") for r in reasonings]
    logger.debug("Conjectures reasoning: %s", conjectures_reasoning)
    # Sometimes additional reasoning is provided at the end, this is okay
    seen_single = False
    for c in conjectures_reasoning:
        if len(c) == 1:
            seen_single = True
        elif len(c) > 1 and seen_single:
            logger.warning("Additional conjecture followed on single reasoning.")
            logger.info("Additional conjecture: %s", c)
            logger.debug("Full LLM output: %s", result)
            break
    conjectures = [c[1] for c in conjectures_reasoning if len(c) > 1]
    conjectures = ["### Conjecture" + c for c in conjectures]
    conjectures = [c.strip() for c in conjectures]
    logger.debug("Conjectures: %s", conjectures)
    return conjectures


def main():
    with open("/home/simon/PycharmProjects/diophantineequations/imo1988q6problem.txt") as file:
        problem = file.read()
    with open("/home/simon/PycharmProjects/diophantineequations/imo1988q6.txt") as file:
        proof = file.read()
    print(generate_conjectures(problem, proof))


if __name__ == '__main__':
    main()
