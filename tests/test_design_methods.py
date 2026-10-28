"""Design method choice (Arya 27 Oct): parametric now; topology and computational listed honestly as planned."""
from physics.design_methods import resolve, DESIGN_METHODS


def test_parametric_is_default_and_available():
    assert resolve(None) == ("parametric", None)
    assert resolve("Parametric") == ("parametric", None)


def test_planned_methods_are_refused_with_a_reason():
    for k in ("topology", "computational"):
        key, err = resolve(k)
        assert key == k and "not available yet" in err and DESIGN_METHODS[k]["stage"] in err


def test_unknown_method():
    key, err = resolve("magic")
    assert key is None and "unknown design_method" in err
