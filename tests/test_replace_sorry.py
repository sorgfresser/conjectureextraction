from diophantineequations.utils import replace_sorry


def test_simple():
    theorem = """theorem test_simple (p : Prop) (hp : p) : p := by sorry"""
    proof = """exact hp"""
    groundtruth = """theorem test_simple (p : Prop) (hp : p) : p := by 
exact hp"""
    assert replace_sorry(theorem, proof) == groundtruth


def test_comments():
    theorem = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
  (h : ∀ x, a * (x^2) + b * x + c = 0) : 
  (b^2 - 4 * a * c) ≥ 0 := 
by
  -- Proof sketch goes here; currently using `sorry` since we are not proving it 
  sorry"""
    proof = """apply sorry"""
    groundtruth = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
  (h : ∀ x, a * (x^2) + b * x + c = 0) : 
  (b^2 - 4 * a * c) ≥ 0 := 
by
  
apply sorry"""
    assert replace_sorry(theorem, proof) == groundtruth


def test_additional_tactics():
    theorem = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := 
    by
      -- Proof sketch goes here; currently using `sorry` since we are not proving it
      something
      sorry"""
    proof = """apply sorry"""
    groundtruth = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := 
    by
      something
      apply sorry"""
    assert replace_sorry(theorem, proof) == groundtruth


def test_additional_tactics_multiple():
    theorem = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := 
    by
      -- Proof sketch goes here; currently using `sorry` since we are not proving it
      something
      something2
      sorry"""
    proof = """apply sorry\napply sorry2"""
    groundtruth = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := 
    by
      something
      something2
      apply sorry
      apply sorry2"""
    assert replace_sorry(theorem, proof) == groundtruth


def test_indented():
    theorem = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := by sorry"""
    proof = """apply sorry\nhave a := by\n  exact b\nexact c"""
    groundtruth = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := by
 apply sorry
 have a := by
   exact b
 exact c"""
    assert replace_sorry(theorem, proof) == groundtruth


def test_without_by():
    theorem = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := sorry"""
    proof = """apply sorry\nhave a := by\n  exact b\nexact c"""
    groundtruth = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := by 
 apply sorry
 have a := by
   exact b
 exact c"""
    assert replace_sorry(theorem, proof) == groundtruth


def test_without_by_newline():
    theorem = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 :=
       sorry"""
    proof = """apply sorry\nhave a := by\n  exact b\nexact c"""
    groundtruth = """theorem conjecture_discriminant_non_negative (a b c k : ℝ) 
      (h : ∀ x, a * (x^2) + b * x + c = 0) : 
      (b^2 - 4 * a * c) ≥ 0 := by 
       apply sorry
       have a := by
         exact b
       exact c"""
    assert replace_sorry(theorem, proof) == groundtruth
