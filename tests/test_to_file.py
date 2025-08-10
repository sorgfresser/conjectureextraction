from diophantineequations.models import FormalizedConjecture
from diophantineequations.utils import get_lemma_from_file
from uuid import uuid4


def test_to_file(tmp_path):
    conjecture = FormalizedConjecture("Given p, show p", "theorem show_p (p : Prop) (h : p) : p := by \nexact h",
                                      imports=[])
    val = tmp_path / f"{str(uuid4())}.lean"
    conjecture.to_file(val)
    assert get_lemma_from_file(val) == "theorem <a>show_p</a> (p : Prop) (h : p) : p"


def test_imports(tmp_path):
    conjecture = FormalizedConjecture("Given p, show p", "theorem show_p (p : Prop) (h : p) : p := by \nexact h",
                                      imports=["import Mathlib"])
    val = tmp_path / f"{str(uuid4())}.lean"
    conjecture.to_file(val)
    assert get_lemma_from_file(val) == "theorem <a>show_p</a> (p : Prop) (h : p) : p"


def test_namespace(tmp_path):
    conjecture = FormalizedConjecture("Given p, show p", "theorem show_p (p : Prop) (h : p) : p := by \nexact h",
                                      imports=["import Mathlib"])
    val = tmp_path / f"{str(uuid4())}.lean"
    conjecture.to_file(val, namespace="something")
    assert get_lemma_from_file(val) == "theorem <a>show_p</a> (p : Prop) (h : p) : p"
    with val.open("r") as f:
        data = f.read()
    assert data == "import Mathlib\nnamespace something\ntheorem show_p (p : Prop) (h : p) : p := by \nexact h\nend something"


def test_namespace_uuid(tmp_path):
    conjecture = FormalizedConjecture("Given p, show p", "open Nat\ntheorem show_p (p : Prop) (h : p) : p := by \nexact h",
                                      imports=["import Mathlib"])
    val = tmp_path / f"{str(uuid4())}.lean"
    uuid_str = "a" + str(uuid4()).split("-")[0]
    conjecture.to_file(val, namespace=uuid_str)
    assert get_lemma_from_file(val) == "theorem <a>show_p</a> (p : Prop) (h : p) : p"
    with val.open("r") as f:
        data = f.read()
    assert data == f"import Mathlib\nnamespace {uuid_str}\nopen Nat\ntheorem show_p (p : Prop) (h : p) : p := by \nexact h\nend {uuid_str}"
