"""Incomplete and error task ids still cover the evaluation protocol.

Polyglot reports a failed task in ``incomplete_ids`` or ``error_ids``. Those
ids were submitted and did not resolve, so they belong in unresolved coverage.
An id that appears in none of the outcome lists is still a protocol mismatch,
including when ``submitted_ids`` lists it and ``accuracy_score`` is non-zero.
"""

import json
from pathlib import Path

import pytest

from EvoPilot_outer import build_evidence, snapshot_from_initial_metadata
from evopilot.controller import RecoveryController
from evopilot.types import Decision
from utils.evo_utils import get_all_performance


SMALL = json.loads(Path("polyglot/subsets/small.json").read_text(encoding="utf-8"))
INCOMPLETE = "java__sgf-parsing"
BASELINE_RESOLVED = ("go__dominoes", "rust__variable-length-quantity")


def _write_report(directory, name, payload):
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _overall_from_report(tmp_path, payload, keyword="child"):
    _write_report(tmp_path, f"{keyword}_0.json", payload)
    _performances, overall = get_all_performance(keyword, results_dir=str(tmp_path))
    assert overall is not None
    return overall


def _screen_report(*, resolved, unresolved, empty=(), incomplete=(), error=(), resolved_instances=None, submitted_ids=None):
    resolved = list(resolved)
    unresolved = list(unresolved)
    empty = list(empty)
    incomplete = list(incomplete)
    error = list(error)
    submitted = list(submitted_ids) if submitted_ids is not None else list(SMALL)
    if resolved_instances is None:
        resolved_instances = len(resolved)
    return {
        "submitted_instances": len(submitted),
        "resolved_instances": resolved_instances,
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "empty_patch_ids": empty,
        "incomplete_ids": incomplete,
        "error_ids": error,
        "submitted_ids": submitted,
    }


def _metadata(overall, entry=SMALL[0]):
    return {
        "run_id": "child",
        "entry": entry,
        "model_patch_notempty": True,
        "overall_performance": overall,
    }


def _controller():
    controller = RecoveryController()
    controller.bootstrap(snapshot_from_initial_metadata("initial_polyglot/metadata.json", tuple(SMALL)))
    return controller


def _evidence(controller, overall):
    return build_evidence(
        _metadata(overall),
        controller,
        debug_round=0,
        escalations_used=0,
        already_counted=0,
        patch_path=None,
        protocol_ids=tuple(SMALL),
    )


def _covered(overall):
    return (
        set(overall["total_resolved_ids"])
        | set(overall["total_unresolved_ids"])
        | set(overall["total_emptypatch_ids"])
    )


def test_incomplete_ids_keep_the_screen_comparable(tmp_path):
    """The reported symptom: accuracy 0.1, one incomplete id, union length 9.

    Folding that id into unresolved makes the 10-task screen complete, so the
    0.1 score is comparable with the 0.2 baseline and is not discarded.
    """

    unresolved = [item for item in SMALL if item not in BASELINE_RESOLVED and item != INCOMPLETE]
    resolved = [BASELINE_RESOLVED[0]]
    overall = _overall_from_report(
        tmp_path,
        _screen_report(resolved=resolved, unresolved=unresolved, incomplete=[INCOMPLETE]),
    )
    assert overall["accuracy_score"] == pytest.approx(0.1)
    assert len(set(overall["total_resolved_ids"]) | set(unresolved)) == 9
    assert _covered(overall) == set(SMALL)
    assert INCOMPLETE in overall["total_unresolved_ids"]
    assert overall["total_emptypatch_ids"] == []

    controller = _controller()
    assert controller.active.score == pytest.approx(0.2)
    built = _evidence(controller, overall)
    assert built.protocol_mismatch is False
    assert built.candidate_score == pytest.approx(0.1)
    assert INCOMPLETE in built.unresolved_ids
    result = controller.step(built)
    assert result.reason != "incomparable_evaluation_protocol"
    assert result.decision != Decision.ACCEPT
    assert controller.active.score == pytest.approx(0.2)


def test_id_missing_from_every_outcome_list_stays_incomparable(tmp_path):
    unresolved = [item for item in SMALL if item not in BASELINE_RESOLVED and item != INCOMPLETE]
    overall = _overall_from_report(
        tmp_path,
        _screen_report(
            resolved=list(BASELINE_RESOLVED[:1]),
            unresolved=unresolved,
            resolved_instances=9,
        ),
    )
    report = json.loads((tmp_path / "child_0.json").read_text(encoding="utf-8"))
    assert INCOMPLETE in report["submitted_ids"]
    for key in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "incomplete_ids", "error_ids"):
        assert INCOMPLETE not in report[key]
    assert INCOMPLETE not in _covered(overall)
    assert len(_covered(overall)) == 9
    assert overall["accuracy_score"] == pytest.approx(0.9)

    controller = _controller()
    built = _evidence(controller, overall)
    assert built.protocol_mismatch is True
    assert built.candidate_score is None
    result = controller.step(built)
    assert result.decision == Decision.ROLLBACK
    assert result.reason == "incomparable_evaluation_protocol"
    assert controller.active.score == pytest.approx(0.2)


def test_error_ids_count_as_unresolved_coverage(tmp_path):
    unresolved = [item for item in SMALL if item not in BASELINE_RESOLVED and item != INCOMPLETE]
    overall = _overall_from_report(
        tmp_path,
        _screen_report(
            resolved=[BASELINE_RESOLVED[0]],
            unresolved=unresolved,
            error=[INCOMPLETE],
        ),
    )
    assert INCOMPLETE in overall["total_unresolved_ids"]
    assert overall["total_unresolved_ids"].count(INCOMPLETE) == 1
    assert _covered(overall) == set(SMALL)

    controller = _controller()
    built = _evidence(controller, overall)
    assert built.protocol_mismatch is False
    assert built.candidate_score == pytest.approx(0.1)
    assert INCOMPLETE in built.unresolved_ids
    result = controller.step(built)
    assert result.reason != "incomparable_evaluation_protocol"


def test_empty_patch_ids_stay_empty_when_also_listed_incomplete(tmp_path):
    empty_id = "javascript__bottle-song"
    unresolved = [item for item in SMALL if item not in BASELINE_RESOLVED and item != empty_id]
    overall = _overall_from_report(
        tmp_path,
        _screen_report(
            resolved=list(BASELINE_RESOLVED),
            unresolved=unresolved,
            empty=[empty_id],
            incomplete=[empty_id],
        ),
    )
    assert overall["total_emptypatch_ids"] == [empty_id]
    assert empty_id not in overall["total_unresolved_ids"]
    assert _covered(overall) == set(SMALL)

    controller = _controller()
    built = _evidence(controller, overall)
    assert built.protocol_mismatch is False
    assert built.candidate_score == pytest.approx(0.2)
    assert empty_id in built.empty_patch_ids
    assert empty_id not in built.unresolved_ids
    result = controller.step(built)
    assert result.reason != "incomparable_evaluation_protocol"
    assert result.decision != Decision.ACCEPT


def test_structural_regression_still_rolls_back_with_incomplete_coverage(tmp_path):
    unresolved = [item for item in SMALL if item != INCOMPLETE]
    overall = _overall_from_report(
        tmp_path,
        _screen_report(resolved=[], unresolved=unresolved, incomplete=[INCOMPLETE], resolved_instances=0),
    )
    assert _covered(overall) == set(SMALL)

    controller = _controller()
    built = _evidence(controller, overall)
    assert built.protocol_mismatch is False
    assert built.candidate_score == pytest.approx(0.0)
    assert built.structural_error is True
    result = controller.step(built)
    assert result.decision == Decision.ROLLBACK
    assert result.reason == "structural_or_catastrophic_regression"
    assert controller.active.score == pytest.approx(0.2)


def test_reports_without_failure_id_lists_keep_the_original_partition(tmp_path):
    unresolved = [item for item in SMALL if item not in BASELINE_RESOLVED]
    payload = _screen_report(resolved=list(BASELINE_RESOLVED), unresolved=unresolved)
    del payload["incomplete_ids"]
    del payload["error_ids"]
    overall = _overall_from_report(tmp_path, payload)
    assert overall["total_resolved_ids"] == list(BASELINE_RESOLVED)
    assert overall["total_unresolved_ids"] == unresolved
    assert overall["total_emptypatch_ids"] == []

    controller = _controller()
    built = _evidence(controller, overall)
    assert built.protocol_mismatch is False
    assert built.candidate_score == pytest.approx(0.2)
    result = controller.step(built)
    assert result.reason != "incomparable_evaluation_protocol"
    assert result.decision != Decision.ACCEPT


def test_failure_id_already_classified_is_not_duplicated(tmp_path):
    payload = _screen_report(
        resolved=["a"],
        unresolved=["b"],
        incomplete=["b"],
        error=["b", "c"],
        submitted_ids=["a", "b", "c"],
        resolved_instances=1,
    )
    overall = _overall_from_report(tmp_path, payload, keyword="synth")
    assert overall["total_resolved_ids"] == ["a"]
    assert overall["total_unresolved_ids"] == ["b", "c"]
    assert overall["total_emptypatch_ids"] == []
