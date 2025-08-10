from diophantineequations.models import LeanFile
from pathlib import Path
import pytest
import os


@pytest.fixture
def dirpath():
    return Path(__file__).parent


def test_relative_to_absolute_filepath_absolute(dirpath):
    relative_to = dirpath
    filepath = dirpath / ".lake/packages/mathlib/Mathlib/Data/Nat/GCD/BigOperators.lean"
    lf = LeanFile(filepath, "")
    assert lf.import_string(relative_to) == "import Mathlib.Data.Nat.GCD.BigOperators\n"


def test_relative_to_absolute_filepath_relative(dirpath):
    relative_to = dirpath
    filepath = Path(".lake/packages/mathlib/Mathlib/Data/Nat/GCD/BigOperators.lean")
    lf = LeanFile(filepath, "")
    assert lf.import_string(relative_to) == "import Mathlib.Data.Nat.GCD.BigOperators\n"


def test_relative_to_relative_filepath_relative():
    relative_to = Path(".")
    filepath = Path(".lake/packages/mathlib/Mathlib/Data/Nat/GCD/BigOperators.lean")
    lf = LeanFile(filepath, "")
    assert lf.import_string(relative_to) == "import Mathlib.Data.Nat.GCD.BigOperators\n"


def test_relative_to_relative_filepath_absolute(dirpath):
    relative_to = Path(".")
    filepath = dirpath / ".lake/packages/mathlib/Mathlib/Data/Nat/GCD/BigOperators.lean"
    lf = LeanFile(filepath, "")

    prefix_str = ""
    if relative_to.resolve() != dirpath:
        prefix = dirpath.relative_to(relative_to.resolve())
        prefix_str = str(prefix).replace("/", ".") + "."

    assert lf.import_string(relative_to) == f"import {prefix_str}Mathlib.Data.Nat.GCD.BigOperators\n"
