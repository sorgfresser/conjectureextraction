from diophantineequations.reprover import _remove_premise_suffix

def test_remove_premise_suffix_no_suffix():
    tactic = "rfl"
    assert _remove_premise_suffix(tactic) == tactic

def test_remove_premise_suffix_with_whitespace():
    tactic = "rw [sq]"
    assert _remove_premise_suffix(tactic) == tactic


def test_remove_premise_suffix_with_colon():
    tactic = "obtain ⟨k, hk⟩ := hdiv"
    assert _remove_premise_suffix(tactic) == tactic

def test_multiple_whitespaces():
    tactic = "rw [← pow_two, ← pow_two]"
    assert _remove_premise_suffix(tactic) == tactic

def test_suffix():
    tactic = "rcases conjecture_a_b_exists_k</a> a b with ⟨k, rfl⟩"
    assert _remove_premise_suffix(tactic) == "rcases conjecture_a_b_exists_k a b with ⟨k, rfl⟩"

def test_broken_suffix():
    tactic = "rcases conjecture_a_b_exists_k</a a b h with ⟨k, rfl⟩"
    assert _remove_premise_suffix(tactic) == "rcases conjecture_a_b_exists_k a b h with ⟨k, rfl⟩"
