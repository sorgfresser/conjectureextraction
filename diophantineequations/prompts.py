GEN_CONTEXT = """You are given a natural language problem statement and a natural language proof for a problem stated in an undergraduate mathematics competition."""

CONJECTURE_GEN = """Decompose this proof into useful smaller conjectures that once proven will aid to prove the full statement.
Do not try to prove the created conjectures, only focus on generating the best conjectures possible.
Good conjectures are strong enough to be useful for the overall proof, while being easier to prove than the full proof.

Please give me some reasoning for before generating a conjecture, so I can understand your thought process. 
Please state them one by one, while adhering to the following format:
**Reasoning:** 
<reasoning>

### Conjecture
<Conjecture-Content>"""

FORMALIZATION_CONTEXT = """You are a math and Lean 4 expert. You are given a natural language problem statement, a natural language proof, and the same problem statement formalized in Lean 4 for a problem stated in an undergraduate mathematics competition.
Furthermore, you are provided with a natural language conjecture that would be helpful for the overall proof, if proven."""

FORMALIZATION_FEW_SHOT = """For example, the conjecture
### Conjecture Quadratic Equation Formation

Given positive integers \( a \), \( b \), \( k \), where \( a^2 + b^2 = k(ab + 1) \). Show that the equation \( a^2 - k b a + b^2 - k = 0 \) holds.

Can be formalized as
```lean4
theorem conjecture_quadratic_equation_formation (a b k : Nat) (h : a > 0 ∧ b > 0 ∧ k > 0) (hk : a^2 + b^2 = k * (a * b + 1)) :
    a^2 - k * b * a + b^2 - k = 0 := by
    -- Normalize hypotheses
    aesop
    -- So we can do subtraction properly, we need the fact that k and k * b * a > 0 respectively
    have h2: k * b * a > 0 := by aesop
    -- Normalize
    ring_nf
    sorry
```
"""

CONJECTURE_FORMALIZATION = """Formalize this natural language conjecture in Lean 4.
You are NOT supposed to prove it, only formalize the statement in Lean 4 syntax.
Feel free to include the start of a proof though. Simply end on by sorry at some point, I will continue from there.
You are not obliged to do a full proof - rather than entering something incorrect, just put a sorry.
"""


CONJECTURE_PROOF = """You are given a Lean 4 statement and the corresponding proof state at the `by sorry`.
Write a proof following Lean 4 syntax for this proof state.
Here are some important syntax changes from Lean 3 to Lean 4 - make sure you use the Lean 4 versions!
- Enter tactic mode by using `by` and then a newline, and indenting the tactics.
  example : True := by
    trivial
- Don't place comma's after tactics, you can go to the next tactic if you write it on a new line (in the same column)
- Function notation changed from λ x, f x to fun x => f x or λ x => f x or fun x ↦ f x or λ x ↦ f x
- Π x : A, P x is not legal anymore. Use ∀ x : A, P x or (x : A) → P x
- The square brackets in rw [h] are mandatory.
- split has been removed. Use constructor.
- Use `variable` to introduce multiple variables as well, there is no separate `variables`
New naming conventions:
- Terms of Props (e.g. proofs, theorem names) are in snake_case.
- Props and Types (or Sort) (inductive types, structures, classes) are in UpperCamelCase.
- Anything else is in lowerCamelCase.
New tactic syntax
- cases syntax changed, new syntax is with | and =>
example (h : p ∨ q) : q ∨ p := by
  cases h with
  | inl hp => exact Or.inr hp
  | inr hq => exact Or.inl hq
  
Make sure to use this new syntax! Whatever you do, always guarantee you are using the new Lean 4 syntax.
It is of utmost importance!
Only provide the lean output starting with `by`. Do not output the whole statement again!
Give me an overall analysis of the proof state and what is to do first, then prove.
"""

CONJECTURE_PROOF_BYSTART = "Your generated lean code does not start with by - maybe it restates the theorem? Avoid this at all cost. Make sure your lean code block starts with by!"


CONJECTURE_NL_FALSE = """You are given a natural language conjecture statement and the corresponding formalized Lean 4 conjecture.
The formalized version differs in one important aspect: the conclusion is replaced with false, while all the assumptions remain the same. 
Now also adjust the natural language statement to only conclude false, without changing any of the assumptions. 
Only return the natural language statement! Please return the full natural language statement with your adjustments, not just the edited sections."""

CONJECTURE_NL_FALSE_FEW_SHOT = """For example, the following natural language conjecture:
### Conjecture 1
Given a logical framework with the assumption \\( h: \\text{False} \\), assume some universally true statement \\( X \\).
Shows that assuming \\( h \\) implies a result \\( Y \\) that contradicts \\( X \\).

would be adjusted to:
### Conjecture 1
Given: A logical framework with the assumption \\( h: \\text{False} \\), assume some universally true statement \\( X \\).
Show False.
"""

CONJECTURE_EVALUATION = """You are given a conjecture statement in natural language and formalized In Lean 4.
Additionally, you receive the final theorem that I would like to proof in Lean 4 using this conjecture.
Please, give me an indication to how relevant this conjecture statement is with respect to the final theorem.
End your analysis by giving me a number in the following format:
Relevance: [0-10]
"""


INFORMAL_PROOF = """You are given a conjecture statement in natural language and formalized in Lean 4.
The given conjecture is a crucial part of the proof for a larger theorem. I will provide you with the full theorem statement.
Your job is to give me a natural language proof for the conjecture.
Do not implement the proof in Lean, just provide me with a textual proof."""

INFORMALIZATION = """You are given a Lean 4 file. Your task is to provide a natural language description of the file.
In particular, you should describe the main definitions of the file. Also provide a brief overview of the main theorems and lemmas in the file.
Please provide a high-level overview, do not go into the specifics of the proofs."""

INFORMALIZATION_FEW_SHOT = """For example, the following Lean 4 code:
```lean
import Mathlib.Algebra.Quaternion -- probably get away with less

-- NOTE: this is all in/on the way to mathlib.

/-!
# Characteristic predicate for central simple algebras

In this file we define the predicate `IsCentralSimple K D` where `K` is a field
and `D` is a (noncommutative) `K`-algebra.

Note that the predicate makes sense just for `K` a `CommRing` but it doesn't give the
right definition; for a commutative ring base one should use the theory of Azumaya algebras.
This adds an extra layer of complication which we don't need. In fact ideals of `K`
immediately give rise to nontrivial quotients of `D` so there are no central simple
algebras in this case according to our definition.

-/

universe u v w

open Classical
open scoped BigOperators

structure IsCentralSimple
    (K : Type u) [Field K] (D : Type v) [Ring D] [Algebra K D] : Prop where
  is_central : ∀ d : D, d ∈ Subring.center D → ∃ k : K, d = algebraMap K D k
  is_simple : IsSimpleOrder (RingCon D)

variable (K : Type u) [Field K]

theorem RingCon.sum {R : Type u} [AddCommMonoid R] [Mul R] {ι : Type v} {s : Finset ι} {a b : ι → R}
    {r : RingCon R} (h : ∀ i ∈ s, r (a i) (b i)) : r (∑ i in s, a i) (∑ i in s, b i) := by
  induction s using Finset.induction_on with
  | empty =>
    simp only [Finset.sum_empty]
    exact r.refl 0
  | insert hj ih =>
    next h' j s' =>
      simp_rw [Finset.sum_insert hj]
      apply RingCon.add
      · exact h j (Finset.mem_insert_self j s')
      · exact ih fun i hi ↦ h i (Finset.mem_insert_of_mem hi)

open Matrix in
theorem MatrixRing.isCentralSimple (ι : Type v) (hι : Fintype ι) [Nonempty ι] [DecidableEq ι] :
    IsCentralSimple K (Matrix ι ι K) where
  is_central d hd := by
    rw [Subring.mem_center_iff] at hd
    convert mem_range_scalar_of_commute_stdBasisMatrix (M := d) fun i j _ => hd _
    simp_rw [Set.mem_range, eq_comm, algebraMap_eq_diagonal, Pi.algebraMap_def,
      Algebra.id.map_eq_self, scalar_apply]
  is_simple.eq_bot_or_eq_top := by
    intro r
    obtain h | h := _root_.forall_or_exists_not (fun x ↦ r 0 x ↔ x = 0)
    · left
      apply RingCon.ext
      intro x y
      have : r x y ↔ r 0 (y - x) := by
        constructor
        · convert RingCon.add r (r.refl (-x)) using 1
          rw [neg_add_cancel, sub_eq_add_neg, add_comm]
        · convert RingCon.add r (r.refl x) using 1
          rw [add_sub_cancel, add_zero]
      rw [this, h, sub_eq_zero, eq_comm, RingCon.coe_bot]
    · right
      obtain ⟨x, hx⟩ := h
      have x_ne_zero : x ≠ 0 := by
        rintro rfl
        simp [eq_true (r.refl 0)] at hx
      have r_zero_x : r 0 x := by tauto
      have : ∃ i j, x i j ≠ 0 := by simpa using x_ne_zero ∘ Matrix.ext
      obtain ⟨i, j, hij⟩ := this
      have (k : ι) (_ : k ∈ Finset.univ) :
          r 0 ((stdBasisMatrix k i 1) * x * (stdBasisMatrix j k 1)) := by
        simpa using
          r.mul (r.mul (r.refl (stdBasisMatrix k i 1)) r_zero_x) (r.refl (stdBasisMatrix j k 1))
      have r_zero_sum := RingCon.sum this
      have sum_eq_scalar :
          ∑ k, (stdBasisMatrix k i 1) * x * (stdBasisMatrix j k 1) = scalar ι (x i j) := by
        ext i' j'
        simp [diagonal, sum_apply, mul_apply, stdBasisMatrix, ite_and, eq_comm]
      have r_zero_one : r 0 1 := by
        simpa [hij, Finset.sum_const_zero, sum_eq_scalar] using
          r.mul r_zero_sum (r.refl (scalar ι (x i j)⁻¹))
      have forall_r_zero a : r 0 a := by simpa using r.mul r_zero_one (r.refl a)
      have forall_forall_r a b : r a b := by simpa using r.add (forall_r_zero (b - a)) (r.refl a)
      apply RingCon.ext
      simp [forall_forall_r]

namespace IsCentralSimple

variable (D : Type v) [Ring D] [Algebra K D] (h : IsCentralSimple K D)

/-
\begin{lemma}
    \label{IsCentralSimple.baseChange}
    If $D$ is a central simple algebra over~$K$ and $L/K$ is a field extension, then $L\otimes_KD$
    is a central simple algebra over~$L$.
\end{lemma}
\begin{proof}
    This is not too hard: it's lemma b of section 12.4 in Peirce's "Associative algebras".
    Will maybe write more on Saturday.
\end{proof}
-/

open scoped TensorProduct

-- lemma baseChange (L : Type w) [Field L] [Algebra K L] : IsCentralSimple L (L ⊗[K] D) := sorry

end IsCentralSimple

-- restrict to 4d case
-- theorem exists_quaternionAlgebra_iso (hK : (2 : K) ≠ 0) :
--     ∃ a b : K, Nonempty (D ≃ₐ[K] ℍ[K, a, b]) := sorry
 to vector store
INFO:__main__:Adding single file ../FLT/FLT/NumberField/AdeleRing.lean to vector store
DEBUG:__main__:Potential matches for definition: import Mathlib
import FLT.Mathlib.NumberTheory.NumberField.Basic
import FLT.Mathlib.RingTheory.DedekindDomain.AdicValuation

universe u

section LocallyCompact

-- see https://github.com/smmercuri/adele-ring_locally-compact
-- for a proof of this

variable (K : Type*) [Field K] [NumberField K]

instance NumberField.AdeleRing.locallyCompactSpace : LocallyCompactSpace (AdeleRing K) :=
  sorry -- issue #253

end LocallyCompact

section BaseChange

end BaseChange

section Discrete

open NumberField DedekindDomain

theorem Rat.AdeleRing.zero_discrete : ∃ U : Set (AdeleRing ℚ),
    IsOpen U ∧ (algebraMap ℚ (AdeleRing ℚ)) ⁻¹' U = {0} := by
  use {f | ∀ v, f v ∈ (Metric.ball 0 1)} ×ˢ
    {f | ∀ v , f v ∈ IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers ℚ v}
  refine ⟨?_, ?_⟩
  · dsimp
    sorry -- issue #252 -- should be easy (product of opens is open, product of integers is surely
          -- known to be open)
  · apply subset_antisymm
    · intro x hx
      rw [Set.mem_preimage] at hx
      simp only [Set.mem_singleton_iff]
      have : (algebraMap ℚ (AdeleRing ℚ)) x =
        (algebraMap ℚ (InfiniteAdeleRing ℚ) x, algebraMap ℚ (FiniteAdeleRing (𝓞 ℚ) ℚ) x)
      · rfl
      rw [this] at hx
      clear this
      rw [Set.mem_prod] at hx
      obtain ⟨h1, h2⟩ := hx
      dsimp only at h1 h2
      simp only [Metric.mem_ball, dist_zero_right, Set.mem_setOf_eq,
        InfiniteAdeleRing.algebraMap_apply, UniformSpace.Completion.norm_coe] at h1
      simp only [Set.mem_setOf_eq] at h2
      specialize h1 Rat.infinitePlace
      change ‖(x : ℂ)‖ < 1 at h1
      simp at h1
      have intx: ∃ (y:ℤ), y = x
      · obtain ⟨z, hz⟩ := IsDedekindDomain.HeightOneSpectrum.mem_integers_of_valuation_le_one
            (𝓞 ℚ) ℚ x <| fun v ↦ by
          specialize h2 v
          letI : UniformSpace ℚ := v.adicValued.toUniformSpace
          rw [IsDedekindDomain.HeightOneSpectrum.mem_adicCompletionIntegers] at h2
          rwa [← IsDedekindDomain.HeightOneSpectrum.valuedAdicCompletion_eq_valuation']
        use Rat.ringOfIntegersEquiv z
        rw [← hz]
        apply Rat.ringOfIntegersEquiv_eq_algebraMap
      obtain ⟨y, rfl⟩ := intx
      simp only [abs_lt] at h1
      norm_cast at h1 ⊢
      -- We need the next line because `norm_cast` is for some reason producing a `negSucc 0`.
      -- I haven't been able to isolate this behaviour even in a standalone lemma.
      -- We could also make `omega` more robust against accidental appearances of `negSucc`.
      rw [Int.negSucc_eq] at h1
      omega
    · intro x
      simp only [Set.mem_singleton_iff, Set.mem_preimage]
      rintro rfl
      simp only [map_zero]
      change (0, 0) ∈ _
      simp only [Prod.mk_zero_zero, Set.mem_prod, Prod.fst_zero, Prod.snd_zero]
      constructor
      · simp only [Metric.mem_ball, dist_zero_right, Set.mem_setOf_eq]
        intro v
        have : ‖(0:InfiniteAdeleRing ℚ) v‖ = 0
        · simp only [norm_eq_zero]
          rfl
        simp [this, zero_lt_one]
      · simp only [Set.mem_setOf_eq]
        intro v
        apply zero_mem

-- Maybe this discreteness isn't even stated in the best way?
-- I'm ambivalent about how it's stated
open Pointwise in
theorem Rat.AdeleRing.discrete : ∀ q : ℚ, ∃ U : Set (AdeleRing ℚ),
    IsOpen U ∧ (algebraMap ℚ (AdeleRing ℚ)) ⁻¹' U = {q} := by
  obtain ⟨V, hV, hV0⟩ := zero_discrete
  intro q
  set ι  := algebraMap ℚ (AdeleRing ℚ)    with hι
  set qₐ := ι q                           with hqₐ
  set f  := Homeomorph.subLeft qₐ         with hf
  use f ⁻¹' V, f.isOpen_preimage.mpr hV
  have : f ∘ ι = ι ∘ Homeomorph.subLeft q := by ext; simp [hf, hqₐ]
  rw [← Set.preimage_comp, this, Set.preimage_comp, hV0]
  ext
  simp only [Set.mem_preimage, Homeomorph.subLeft_apply, Set.mem_singleton_iff, sub_eq_zero, eq_comm]


variable (K : Type*) [Field K] [NumberField K]

theorem NumberField.AdeleRing.discrete : ∀ k : K, ∃ U : Set (AdeleRing K),
    IsOpen U ∧ (algebraMap K (AdeleRing K)) ⁻¹' U = {k} := sorry -- issue #257

end Discrete

section Compact

open NumberField

theorem Rat.AdeleRing.cocompact :
    CompactSpace (AdeleRing ℚ ⧸ AddMonoidHom.range (algebraMap ℚ (AdeleRing ℚ)).toAddMonoidHom) :=
  sorry -- issue #258

variable (K : Type*) [Field K] [NumberField K]

theorem NumberField.AdeleRing.cocompact :
    CompactSpace (AdeleRing K ⧸ AddMonoidHom.range (algebraMap K (AdeleRing K)).toAddMonoidHom) :=
  sorry -- issue #259

end Compact
```

could be described as:

The provided Lean code is about studying a special type of algebra called *central simple algebras* over a field. Here's an informal breakdown of what's happening:

1. **Central Simple Algebras:**  
   The code defines what it means for an algebra over a field to be *central* and *simple*:  
   - **Central part:** Every element in the center (the set of elements that commute with everything) must be just a scalar multiple of the identity element. Essentially, the only "special" elements in the algebra should come directly from the underlying field.  
   - **Simple part:** The algebra doesn’t have any "non-trivial" two-sided ideals, meaning you can’t break it down into smaller pieces using normal algebraic structures.

2. **Working with Matrix Algebras:**  
   There’s a proof showing that square matrices over a field form an example of such an algebra. The idea is that if you take a matrix that commutes with all others, it must be a scalar multiple of the identity matrix. Also, it shows that the only way to split the structure is either by considering everything or nothing, which confirms its simplicity.

3. **Summing Elements with a Certain Property:**  
   A helper result states that if a certain property holds for each term in a sum, then it also holds for the entire sum. This is useful when dealing with algebraic structures that are built by combining smaller parts.
"""