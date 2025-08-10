from pathlib import Path
import subprocess
import logging
from uuid import uuid4
import weave

from diophantineequations.models import BaseConjecture
from diophantineequations.utils import get_lemma_from_file
import fcntl
import os
import contextlib

logger = logging.getLogger(__name__)


def lake_build(root_path: Path):
    result = subprocess.run(["lake", "build"], cwd=root_path, capture_output=True, encoding="utf-8")
    if result.returncode != 0:
        logger.error("Lake build failed, stderr:\n%s", result.stderr)
        logger.warning("Lake build failed, stdout:\n%s", result.stdout)
    result.check_returncode()

@contextlib.contextmanager
def locked_fd(path, flags):
    fd = os.open(path, flags)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

@weave.op()
def add_conjecture(root_path: Path, src_path: Path, base_file: Path, conjecture: BaseConjecture, wrap_namespace: bool = False) -> Path:
    """Add a conjecture to the given lean project. Will create a new lean file in the lean project src path.
    If the new file leads to a build error (i.e. lake build fails), will revert all changes

    :param root_path: Path to the lean project's root
    :param src_path: Directory inside the root path where all lean src files will be stored
    :param base_file: The path in the lean project's root where the import of the new conjecture will be placed
    :param conjecture: The conjecture to add
    :param wrap_namespace: Whether to encapsulate the conjecture in a randomly generated namespace
    :return: The file path to the newly created conjecture
    """
    # Lock the src path to prevent multiple processes from adding a conjecture with same index at the same time
    uuid_str = "a" + str(uuid4()).split("-")[0]
    with locked_fd(src_path.resolve(), os.O_RDONLY):
        file_indices = [int(str(file.stem).split("_")[1]) for file in src_path.glob("*.lean")]
        curr_idx = max(file_indices) + 1 if file_indices else 0
        filename = f"Conjecture_{curr_idx}.lean"
        new_filepath = src_path / filename
        with locked_fd(new_filepath.resolve(), os.O_CREAT):
            namespace = uuid_str if wrap_namespace else None
            lean_file = conjecture.to_file(new_filepath, namespace=namespace)
        try:
            get_lemma_from_file(lean_file.filepath)
        except AssertionError:
            logger.exception("Failed to get lemma from lean file")
            new_filepath.unlink(missing_ok=False)
            return new_filepath

    with locked_fd(base_file.resolve(), os.O_RDONLY):
        # Keep old conjecture import of base file if we need to revert
        with base_file.open("r") as file:
            data = file.read()
        # Prepend
        import_string = lean_file.import_string(base_file.parent)
        prepended_data = import_string + data
        # full_data = data + "\n" + lean_file.content
        with base_file.open("w") as file:
            # file.write(full_data)
            file.write(prepended_data)
        try:
            lake_build(root_path)
        except subprocess.CalledProcessError:
            # Revert
            logger.exception("Lake build failed, reverting conjecture")
            with base_file.open("w") as file:
                file.write(data)
            new_filepath.unlink(missing_ok=False)
    return new_filepath

if __name__ == '__main__':
    from diophantineequations.models import FormalizedConjecture
    add_conjecture(
        Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project"),
        Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project/Imo1988q6project"),
        Path("/home/simon/PycharmProjects/diophantineequations/imo1988q6project/Imo1988q6project.lean"),
        FormalizedConjecture(
            "I am a theorem",
            """theorem conjecture_minimality_implies_bound_on_the_other_root (a b k : Nat) (c : Int)
                (h : a > 0 ∧ b > 0 ∧ k > 0 ∧ a ≥ b)
                (hk : a^2 - k * b * a + b^2 - k = 0)
                (hc : c = k * b - a) :
                c ≤ 0 := by
              sorry""",
            [],
        ),
        wrap_namespace=True
    )
