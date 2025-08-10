from diophantineequations.utils import get_lemma_from_file
from pathlib import Path
from io import StringIO
from tempfile import NamedTemporaryFile


def _try_theorem(theorem: str):
    with NamedTemporaryFile(mode="w") as f:
        f.write(theorem)
        f.seek(0)
        return get_lemma_from_file(Path(f.name))


def test_simple():
    to_test = """theorem test_conjecture : 1 = 1 := by sorry"""
    assert _try_theorem(to_test) == "theorem <a>test_conjecture</a> : 1 = 1"


def test_theorem_name():
    to_test = """theorem test_theorem : 1 = 1 := by sorry"""
    assert _try_theorem(to_test) == "theorem <a>test_theorem</a> : 1 = 1"


def test_theorem_with_colon_equals():
    to_test = """theorem conjecture_a_b_exists_k (a b : ℕ) (h : (a * b + 1) ∣ (a * a + b * b)) : ∃ k : ℤ, a * a + b * b = k * (a * b + 1) := by
obtain ⟨k, hk⟩ := h
use k
ring
norm_cast
rw [sq]
rw [sq]
rw [hk]
ring"""
    assert _try_theorem(
        to_test) == "theorem <a>conjecture_a_b_exists_k</a> (a b : ℕ) (h : (a * b + 1) ∣ (a * a + b * b)) : ∃ k : ℤ, a * a + b * b = k * (a * b + 1)"


def test_by_in_theorem_name():
    to_test = """theorem conjecture_dividing_by_a2 (a b k : ℕ) (h1 : a = b) (h2 : a > 0) (h3 : a^2 * (2 - k) = k) :
    2 - k = k / a^2 := by
rw [← h3]
rw [h1]
rw [← h3]
rw [h1]
rw [← h1]
rw [← h3]
rw [h1]
rw [← h3]
rw [h1]
rw [← h1]
field_simp
congr"""
    assert _try_theorem(to_test) == """theorem <a>conjecture_dividing_by_a2</a> (a b k : ℕ) (h1 : a = b) (h2 : a > 0) (h3 : a^2 * (2 - k) = k) :
    2 - k = k / a^2"""


def test_by_in_imports():
    to_test = """import Mathlib.NumberTheory.SumTwoSquares
import Mathlib.NumberTheory.SumFourSquares
import Mathlib.Algebra.Ring.Identities
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.NumberTheory.FLT.Four
import Mathlib.Algebra.Ring.Int.Parity
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Data.by
import Mathlib.Algebra.Group.Nat.Even
theorem conjecture_dividing_a2 (a b k : ℕ) (h1 : a = b) (h2 : a > 0) (h3 : a^2 * (2 - k) = k) :
    2 - k = k / a^2 := by
rw [← h3]"""
    assert _try_theorem(to_test) == """theorem <a>conjecture_dividing_a2</a> (a b k : ℕ) (h1 : a = b) (h2 : a > 0) (h3 : a^2 * (2 - k) = k) :
    2 - k = k / a^2"""


def test_minif2f():
    to_test = """theorem mathd_numbertheory_447 :
∑ k ∈ Finset.filter (λ x => 3∣x) (Finset.Icc 1 49), (k % 10) = 78 := by aesop 
"""
    assert _try_theorem(
        to_test) == """theorem <a>mathd_numbertheory_447</a> :\n∑ k ∈ Finset.filter (λ x => 3∣x) (Finset.Icc 1 49), (k % 10) = 78"""

    to_test = """theorem mathd_algebra_125
(x y : ℕ)
(h₀ : 0 < x ∧ 0 < y)
(h₁ : 5 * x = y)
(h₂ : (↑x - (3:ℤ)) + (y - (3:ℤ)) = 30) :
x = 6 := by
linarith
"""
    assert _try_theorem(to_test) == """theorem <a>mathd_algebra_125</a> (xy : ℕ)
(h₀ : 0 < x ∧ 0 < y)
(h₁ : 5 * x = y)
(h₂ : (↑x - (3:ℤ)) + (y - (3:ℤ)) = 30) :
x = 6"""
