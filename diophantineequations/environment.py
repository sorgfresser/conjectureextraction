from logging import getLogger
from pathlib import Path

from lean_repl_py import LeanREPLHandler

from diophantineequations.utils import text_without_comments

logger = getLogger(__name__)

LEAN4_DEFAULT_HEADER = "import Mathlib.Tactic.Linarith\nimport Mathlib.Tactic.Ring.RingNF\nimport Mathlib.Tactic.FieldSimp\nimport Mathlib.Tactic.Ring\nimport Mathlib.Tactic.LinearCombination"
DEEPSEEK_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

def _import_handler(handler: LeanREPLHandler, deepseek: bool = False):
    header = DEEPSEEK_DEFAULT_HEADER if deepseek else LEAN4_DEFAULT_HEADER
    logger.info("Sending command: %s", header)
    handler.send_command(header)
    received_tuple = handler.receive_json()
    logger.info("Received tuple: %s", received_tuple)
    assert received_tuple is not None
    _, env = received_tuple
    handler.env = env.env_index
    return handler

def get_imports(code: str) -> list[str]:
    """Get all import statements from the given code."""
    text = text_without_comments(code)
    imports = [line for line in text.split("\n") if line.strip().startswith("import")]
    logger.debug("Imports: %s", imports)
    return imports

def import_code(handler: LeanREPLHandler, code: str, deepseek: bool = False) -> LeanREPLHandler:
    """Execute all import statements in the given code.

    Will reset the env of the handler to this new env with the correct imports.
    :param handler: Lean REPL handler to use
    :param code: The Lean code whose imports to load
    :param deepseek: Whether deepseek is used or not
    :return: The modified handler
    """
    imports = get_imports(code)
    if not imports:
        logger.info("Did not find any import statements, returning")
        return handler
    # We can only use a single command with imports in the repl, so we concat previous ones
    command = "\n".join(imports)
    header = LEAN4_DEFAULT_HEADER if not deepseek else DEEPSEEK_DEFAULT_HEADER
    command = header + "\n" + command
    # Resetting env as the current env already has imports
    handler.env = None
    logger.info("Executing in import code: %s", command)
    handler.send_command(command)
    received_tuple = handler.receive_json()
    logger.info("Received tuple: %s", received_tuple)
    assert received_tuple is not None
    _, env = received_tuple
    handler.env = env.env_index
    return handler


def import_file(handler: LeanREPLHandler, filepath: Path) -> LeanREPLHandler:
    """
    Execute all import statements in the given file. Will set the env of the handler accordingly
    :param handler: Lean REPL handler to use
    :param filepath: The file whose imports to load
    :return: The modified handler
    """
    logger.debug("Importing file %s", filepath.absolute())
    with filepath.open("r") as file:
        data = file.read()
        logger.debug("Read file %s, content: %s", filepath.absolute(), data)
    return import_code(handler, data)


def get_handler(root_path: Path, deepseek: bool = False) -> LeanREPLHandler:
    handler = LeanREPLHandler(root_path)
    handler = _import_handler(handler, deepseek)
    return handler
