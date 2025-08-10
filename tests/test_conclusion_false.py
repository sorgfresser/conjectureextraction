from diophantineequations.utils import conclusion_false


def test_simple():
    theorem = """theorem conjecture_minimality_implies_bound_on_the_other_root (a b k : Nat) (c : Int)
    (h : a > 0 ∧ b > 0 ∧ k > 0 ∧ a ≥ b)
    (hk : a^2 - k * b * a + b^2 - k = 0)
    (hc : c = k * b - a) :
    c ≤ 0 := by
  sorry"""
    groundtruth = """theorem conjecture_minimality_implies_bound_on_the_other_root (a b k : Nat) (c : Int)
    (h : a > 0 ∧ b > 0 ∧ k > 0 ∧ a ≥ b)
    (hk : a^2 - k * b * a + b^2 - k = 0)
    (hc : c = k * b - a) :
    False := by
  sorry"""
    assert conclusion_false(theorem) == groundtruth


def test_exists_nested_colon():
    theorem = """theorem conjecture_root_zero (a b k : Nat) (h : a > 0 ∧ b > 0 ∧ k > 0) (hk : a^2 + b^2 = k * (a * b + 1)) :
    (∃ x : ℕ, x^2 - k * b * x + (b^2 - k) = 0 → x = 0) := by
    sorry"""
    groundtruth = """theorem conjecture_root_zero (a b k : Nat) (h : a > 0 ∧ b > 0 ∧ k > 0) (hk : a^2 + b^2 = k * (a * b + 1)) :
    False := by
    sorry"""
    assert conclusion_false(theorem) == groundtruth


def test_false_already():
    theorem = """theorem conjecture_false (h : False) : False := by
  sorry"""
    assert conclusion_false(theorem) == theorem


def test_exists_without_nesting():
    theorem = """theorem conjecture_existence_integer_k (a b : Nat) (h : a > 0 ∧ b > 0) (h_div : (a * b + 1) ∣ (a^2 + b^2)) :
    ∃ k : Nat, a^2 + b^2 = k * (a * b + 1) := by
    sorry"""
    groundtruth = """theorem conjecture_existence_integer_k (a b : Nat) (h : a > 0 ∧ b > 0) (h_div : (a * b + 1) ∣ (a^2 + b^2)) :
    False := by
    sorry"""
    assert conclusion_false(theorem) == groundtruth


def test_colon_equals_nested():
    theorem = """theorem conjecture_quadratic_root (a b k : Int) (h : a > 0 ∧ b > 0 ∧ k > 0) 
    (hk : a^2 + b^2 = k * (a * b + 1)) 
    (b' : Int := min a b) (a' : Int := max a b) :
    ∃ c' : Int, c' ≠ a' ∧ (a'^2 - k * b' * a' + b'^2 - k = 0 ∧ c' ≤ 0) := by
    sorry"""
    groundtruth = """theorem conjecture_quadratic_root (a b k : Int) (h : a > 0 ∧ b > 0 ∧ k > 0) 
    (hk : a^2 + b^2 = k * (a * b + 1)) 
    (b' : Int := min a b) (a' : Int := max a b) :
    False := by
    sorry"""
    assert conclusion_false(theorem) == groundtruth


def test_colon_equals_non_nested():
    theorem = """theorem conjecture_minimized_root (a b k : Nat) (h : a > 0 ∧ b > 0 ∧ k > 0) (hk : a^2 + b^2 = k * (a * b + 1)) :
    let a' := max a b
    let b' := min a b
    a'^2 - k * b' * a' + (b'^2 - k) = 0 → ∃ c' : ℤ, c' ≤ 0 := by
    sorry"""
    groundtruth = """theorem conjecture_minimized_root (a b k : Nat) (h : a > 0 ∧ b > 0 ∧ k > 0) (hk : a^2 + b^2 = k * (a * b + 1)) :
    False := by
    sorry"""
    assert conclusion_false(theorem) == groundtruth

def test_curly_braces():
    theorem = """theorem conjecture_root_bounds (a b k : ℤ) (h : a^2 + b^2 = k * (a * b + 1)) {c : ℤ}
    (hc : c * c - k * b * c + b * b - k = 0) :
    c ≤ b :=
by
  sorry"""
    groundtruth = """theorem conjecture_root_bounds (a b k : ℤ) (h : a^2 + b^2 = k * (a * b + 1)) {c : ℤ}
    (hc : c * c - k * b * c + b * b - k = 0) :
    False :=
by
  sorry"""
    assert conclusion_false(theorem) == groundtruth

def test_comments():
    theorem = """theorem conjecture_alternate_root (a b k : ℤ) (hb : b > 0) 
  (h : a^2 + b^2 = k * (a * b + 1)) : 
  ∀ c : ℤ, 
  (c : ℤ) < b → (c^2 + b^2 = k * (c * b + 1)) → c ≤ 0 := by
  -- proof goes here
  sorry"""
    groundtruth = """theorem conjecture_alternate_root (a b k : ℤ) (hb : b > 0) 
  (h : a^2 + b^2 = k * (a * b + 1)) : 
  False := by
  sorry"""
    assert conclusion_false(theorem) == groundtruth

def test_comments_single_edge_case():
    theorem = """/-- This is ugly
Nice edgecase -/
theorem conjecture_alternate_root (a b k : ℤ) (hb : b > 0) 
  (h : a^2 + b^2 = k * (a * b + 1)) : 
  False := by
  sorry"""
    groundtruth = """theorem conjecture_alternate_root (a b k : ℤ) (hb : b > 0) 
  (h : a^2 + b^2 = k * (a * b + 1)) : 
  False := by
  sorry"""
    assert conclusion_false(theorem) == groundtruth