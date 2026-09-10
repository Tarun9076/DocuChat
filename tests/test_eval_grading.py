from eval.eval_retrieval import contains_required_facts


def test_all_keywords_present():
    assert contains_required_facts("Jensen Huang founded Nvidia in 1993.", ["Jensen Huang", "1993"])


def test_case_insensitive():
    assert contains_required_facts("JENSEN HUANG founded nvidia in 1993.", ["Jensen Huang", "1993"])


def test_missing_keyword_fails():
    assert not contains_required_facts("Nvidia was founded in 1993.", ["Jensen Huang", "1993"])


def test_empty_must_contain_always_passes():
    assert contains_required_facts("anything at all", [])
