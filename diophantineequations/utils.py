import os
from logging import getLogger
import re
from pathlib import Path
from dataclasses import dataclass
import weave

logger = getLogger(__name__)
WS = r"\s+"
MAX_WS = r"^\s*"
BY_WS = r"^\s*by"
SORRY = r"(\s*)sorry(\s*)$"
THEOREM_REGEX = r"\stheorem[\s(]"

IS_OPENMPI = os.getenv("OMPI_COMM_WORLD_SIZE") is not None
IS_INTEL_MPI = os.getenv("I_MPI_MPIRUN") is not None
IS_DISTRIBUTED = IS_OPENMPI or IS_INTEL_MPI

RANK = None if not IS_DISTRIBUTED else int(os.getenv("OMPI_COMM_WORLD_RANK")) if IS_OPENMPI else int(
    os.getenv("PMI_RANK"))
IS_MASTER = RANK is not None and RANK == 0

if IS_DISTRIBUTED:
    logger.info("Running in distributed mode")
    logger.info("Is master: %s", IS_MASTER)


@weave.op()
def text_without_comments(text: str) -> str:
    logger.debug("Text with comments: %s", text)
    lines = text.split("\n")
    # Drop empty lines
    lines = [line for line in lines if line.strip()]
    in_comment = False
    for line_idx, line in enumerate(lines):
        new_line = ""
        for idx in range(len(line)):
            if idx < len(line) - 1 and line[idx:idx + 2] == "/-":
                in_comment = True
            if not in_comment:
                new_line += line[idx]
            elif idx > 0 and line[idx - 1: idx + 1] == "-/":
                in_comment = False
        lines[line_idx] = new_line
    # Drop empty lines
    single_removed = [line[:line.index("--")] if "--" in line else line for line in lines]
    single_removed = [line for line in single_removed if line.strip()]
    text_without = "\n".join(single_removed)
    logger.debug("Text without comments: %s", text_without)
    return text_without


def text_without_imports(text: str) -> str:
    """
    Remove all import statements from the given text
    :param text: The text to remove imports from
    :return: The text without imports
    """
    logger.debug("Text with imports: %s", text)
    lines = text.split("\n")
    without_imports = [line for line in lines if not line.strip().startswith("import")]
    text_without = "\n".join(without_imports)
    logger.debug("Text without imports: %s", text_without)
    return text_without


def _find_leftmost_not_in_parenthesis(text, substr):
    """
    Get the leftmost instance of a substring that is not in a parenthesis
    :param text: The text to search
    :param substr: The substring to search for
    :return: index of the substring, or -1 if not found
    """
    index = -1
    parenthesis_level = 0
    for i in range(len(text) - len(substr) + 1):
        if text[i] == "(" or text[i] == "[" or text[i] == "{":
            parenthesis_level += 1
        elif text[i] == ")" or text[i] == "]" or text[i] == "}":
            parenthesis_level -= 1
        if text[i:i + len(substr)] == substr and parenthesis_level == 0:
            index = i
            break
    return index


def _find_rightmost_not_in_parenthesis(text, substr):
    """
    Get the rightmost instance of a substring that is not in a parenthesis
    :param text: The text to search
    :param substr: The substring to search for
    :return: index of the substring, or -1 if not found
    """
    index = -1
    parenthesis_level = 0
    for i in range(len(text) - len(substr), -1, -1):
        if text[i] == ")":
            parenthesis_level += 1
        elif text[i] == "(":
            parenthesis_level -= 1
        if text[i:i + len(substr)] == substr and parenthesis_level == 0:
            index = i
            break
    return index


def is_valid(conjecture: str) -> bool:
    conjecture = text_without_comments(conjecture)
    rightmost = _find_rightmost_not_in_parenthesis(conjecture, ":=")
    if rightmost == -1:
        return False
    by_split = [conjecture[:rightmost], conjecture[rightmost + 2:]]
    if not re.sub(WS, "", by_split[1]).endswith("sorry"):
        return False
    return True


@weave.op()
def conclusion_false(conjecture: str) -> str:
    conjecture = text_without_comments(conjecture)
    rightmost = _find_rightmost_not_in_parenthesis(conjecture, ":=")
    assert rightmost != -1
    by_split = [conjecture[:rightmost], conjecture[rightmost + 2:]]
    assert re.sub(WS, "", by_split[1]).endswith("sorry")
    # Get the hypothesis and conclusion separately
    leftmost = _find_leftmost_not_in_parenthesis(by_split[0], ":")
    assert leftmost != -1
    hypothesis_split = [by_split[0][:leftmost], by_split[0][leftmost + 1:]]
    conclusion = hypothesis_split[1]
    # Replace conclusion with false, keeping whitespace the same
    assert conclusion.strip()
    left_ws = re.match(MAX_WS, conclusion)
    span = left_ws.span(0)
    assert span[0] == 0
    right_ws = re.match(MAX_WS, conclusion[::-1])  # reversed string = right ws since there is no rmatch
    span = right_ws.span(0)
    assert span[0] == 0  # i.e. starts at end
    conclusion = left_ws.group(0) + "False" + right_ws.group(0)[::-1]
    hypothesis_split[1] = conclusion
    by_split[0] = ":".join(hypothesis_split)
    return ":=".join(by_split)


@weave.op()
def replace_sorry(conjecture: str, proof: str) -> str:
    """
    Replace the by sorry part of the given conjecture with the proof
    :param conjecture: The conjecture containing by sorry
    :param proof: The proof to replace by sorry with
    :return: The conjecture with the proof instead of by sorry
    """
    logger.debug("Replacing sorry in conjecture %s with proof %s", conjecture, proof)
    conjecture = text_without_comments(conjecture)
    rightmost = _find_rightmost_not_in_parenthesis(conjecture, ":=")
    assert rightmost != -1
    by_split = [conjecture[:rightmost], conjecture[rightmost + 2:]]
    # Sanity check
    assert re.sub(WS, "", by_split[1]).endswith("sorry")
    # Now replace the by sorry with the proof
    proof = re.sub(BY_WS, "", proof)
    sorry_match = re.findall(SORRY, by_split[1])[0]
    sorry_indent = len(sorry_match[0].removeprefix("\n"))
    by_split[1] = re.sub(SORRY, "", by_split[1])
    if not re.match(BY_WS, by_split[1]):
        by_split[1] = " by " + by_split[1]
    prooflines = proof.split("\n")
    initial_indent = 0
    for line in prooflines:
        initial_indent = len(re.findall(MAX_WS, line)[0])
        if line.strip():
            break
    to_add = sorry_indent - initial_indent
    for idx in range(len(prooflines)):
        prooflines[idx] = " " * to_add + prooflines[idx]
    by_split[1] = by_split[1] + "\n" + "\n".join(prooflines)
    result = ":=".join(by_split)
    logger.debug("Resulting conjecture: %s", result)
    return result


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


def theorem_span(conjecture: str) -> tuple[int, int]:
    """
    Get the span of the theorem in the given conjecture
    :param conjecture: The conjecture to get the theorem span from, must be without comments
    :return: The span of the theorem
    """
    logger.debug("Getting theorem span from conjecture %s", conjecture)
    rightmost = _find_rightmost_not_in_parenthesis(conjecture, ":=")
    assert rightmost != -1
    by_split = [conjecture[:rightmost], conjecture[rightmost + 2:]]
    assert re.sub(WS, "", by_split[1]).endswith("sorry")  # sanity check
    leftmost = _find_leftmost_not_in_parenthesis(by_split[0], ":=")
    assert leftmost != -1
    return leftmost, rightmost


@weave.op()
def get_lemma_from_file(
        file_path: Path,
):
    logger.debug("Getting lemmas from file %s", file_path)
    # Only allows one lemma rn
    with open(file_path) as f:
        data = f.read()
    data = text_without_comments(data)
    data = text_without_imports(data)
    data = "\n" + data  # we add this for the upcoming regex, to ensure we get the first theorem
    theorems = list(re.finditer(THEOREM_REGEX, data))
    assert len(theorems) == 1
    # Replace the theoremname with a placeholder, because it might contain by
    start, end = theorem_name_span(data)
    replacement_string = "A" * (end - start)
    replaced = data[:start] + replacement_string + data[end:]
    by_idx = _find_leftmost_not_in_parenthesis(replaced, "by")
    block_before = data[:by_idx]
    start_idx = _find_rightmost_not_in_parenthesis(block_before, ":=")
    block_before, block_after = data[:start_idx], data[start_idx + 2:]
    assert block_after.strip().startswith("by")
    theorem_statement = re.split(THEOREM_REGEX, block_before)[1].strip()
    theorem_splits = theorem_statement.split(" ", 1)
    if "\n" in theorem_splits[0]:
        old, to_add = theorem_splits[0].split("\n", 1)
        theorem_splits[0] = old
        theorem_splits[1] = to_add + theorem_splits[1]
    theorem_name, theorem_inner = theorem_splits
    final_theorem_statement = "theorem" + " <a>" + theorem_name.strip() + "</a> " + theorem_inner
    logger.debug("Obtained lemma %s in file %s", final_theorem_statement, file_path)
    return final_theorem_statement
