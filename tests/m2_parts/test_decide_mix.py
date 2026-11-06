"""Checks on the v3_mix adoption rule."""

import pytest

from src.m2_parts.decide_mix import decide, improves, not_worse, render_markdown


def diff(value, low, high):
    return {"value": value, "ci95": {"low": low, "high": high}}


def report(dataset, segm, bbox=None, split="test", reference="v3_10c"):
    d = {"segm": {"AP": segm}}
    if bbox is not None:
        d["bbox"] = {"AP": bbox}
    return {"dataset": dataset, "split": split, "reference": reference,
            "differences_vs_reference": {"v3_mix": d}}


def test_interval_checks():
    assert not_worse(diff(-0.01, -0.05, 0.03)) and not not_worse(diff(-0.04, -0.07, -0.01))
    assert improves(diff(0.02, 0.005, 0.04)) and not improves(diff(0.02, -0.01, 0.04))
    assert not_worse(diff(None, None, None)) is None


@pytest.mark.parametrize("robo,hitl,adopt", [
    (diff(-0.02, -0.08, 0.04), diff(0.05, 0.02, 0.08), True),    # wide but not worse on Roboflow
    (diff(-0.05, -0.09, -0.01), diff(0.05, 0.02, 0.08), False),  # worse on Roboflow
    (diff(0.03, -0.01, 0.07), diff(0.01, -0.01, 0.03), False),   # HITL not distinguishable
])
def test_rule_on_mask_ap(robo, hitl, adopt):
    out = decide(report("roboflow", robo), report("hitl", hitl), hitl_aligned=True)
    assert out["adopt"] is adopt and out["decides_on"] == "segm"
    assert "optimista" in out["warning"]
    assert ("SÍ" if adopt else "NO") in render_markdown(out)


def test_box_is_reported_but_does_not_decide():
    good, bad = diff(0.05, 0.02, 0.08), diff(-0.05, -0.09, -0.01)
    out = decide(report("roboflow", good, bbox=bad), report("hitl", good, bbox=good), hitl_aligned=True)
    assert out["adopt"] and not out["by_iou"]["bbox"]["cumple"]


def test_without_aligned_criterion_only_roboflow_decides():
    wide, flat = diff(-0.02, -0.08, 0.04), diff(0.0, -0.02, 0.02)
    out = decide(report("roboflow", wide), report("hitl", flat))
    assert out["adopt"] and not out["hitl_aligned"] and "no cuenta" in render_markdown(out)
    assert not decide(report("roboflow", diff(-0.05, -0.09, -0.01)), report("hitl", flat))["adopt"]


def test_rejects_wrong_reports():
    good = diff(0.05, 0.02, 0.08)
    with pytest.raises(ValueError, match="'test'"):
        decide(report("roboflow", good, split="valid"), report("hitl", good))
    with pytest.raises(ValueError, match="reference"):
        decide(report("roboflow", good, reference="v2"), report("hitl", good))
    with pytest.raises(ValueError, match="expected a 'hitl'"):
        decide(report("roboflow", good), report("mix", good))
