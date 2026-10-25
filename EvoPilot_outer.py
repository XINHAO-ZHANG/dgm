"""EvoPilot outer loop.

Single active agent, a small trusted checkpoint bank, and an explicit
recovery policy. ``DGM_outer.py`` is unchanged and remains the archive
baseline.

Dry run (no API keys, no Docker, no Polyglot evolution)::

    python3 EvoPilot_outer.py --dry_run

Live Polyglot evolution uses the same self-improve and harness path as DGM
and needs the API keys plus Docker setup described in the README.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import uuid

from evopilot.controller import RecoveryController
from evopilot.evidence import evidence_from_dgm_metadata, with_escalation_fallback
from evopilot.policy import VerifyPolicy
from evopilot.replan import format_debug_statement, format_improve_suffix
from evopilot.reporting import flush_run, load_run
from evopilot.smoke import DRY_RUN_DISCLAIMER, run_smoke
from evopilot.types import AgentSnapshot, Decision, VerificationEvidence
from utils.common_utils import load_json_file

LIVE_DISCLAIMER = (
    "Log of this EvoPilot run. AUEC uses evaluation counts from this process only. "
    "It is not a reported paper metric."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoPilot recovery controller")
    parser.add_argument("--max_generation", type=int, default=80, help="Maximum improve attempts.")
    parser.add_argument("--debug_budget", type=int, default=2, help="Bounded debug rounds R per candidate.")
    parser.add_argument("--bank_size", type=int, default=4, help="Trusted checkpoint bank capacity K (about 3-5).")
    parser.add_argument("--accept_margin", type=float, default=0.0, help="Required score gain over the active agent.")
    parser.add_argument("--catastrophic_drop", type=float, default=0.2, help="Score drop treated as a structural regression.")
    parser.add_argument("--benchmark", choices=["polyglot", "swe"], default="polyglot")
    parser.add_argument("--dry_run", action="store_true", help="Replay a synthetic trace. No model calls and no Docker.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None, help="Resume a previous EvoPilot output directory.")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated evaluations passed through to self_improve.")
    parser.add_argument("--shallow_eval", action="store_true", help="Do not expand from the small subset to the medium subset.")
    parser.add_argument("--eval_threshold", type=float, default=0.4, help="Resolved-rate gate for the medium subset.")
    parser.add_argument(
        "--post_improve_diagnose",
        action="store_true",
        help="Also run DGM's post-improve diagnose. The controller does not require it.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> str:
    if args.continue_from and not args.dry_run:
        return args.continue_from
    if args.output_dir:
        return args.output_dir
    run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    return os.path.join("output_evopilot", run_id)


def load_task_lists(benchmark: str) -> tuple[list, list]:
    root = "./polyglot/subsets" if benchmark == "polyglot" else "./swe_bench/subsets"
    return load_json_file(os.path.join(root, "small.json")), load_json_file(os.path.join(root, "medium.json"))


def prepare_initial_copy(output_dir: str, benchmark: str) -> None:
    destination = os.path.join(output_dir, "initial")
    if os.path.exists(destination):
        return
    source = "initial_polyglot" if benchmark == "polyglot" else "initial"
    if not os.path.exists(source):
        raise RuntimeError(f"Missing {source}/. Prepare the initial evaluation cache before a live run.")
    shutil.copytree(source, destination)


def snapshot_from_initial_metadata(path: str) -> AgentSnapshot:
    metadata = load_json_file(path)
    perf = metadata.get("overall_performance") or {}
    return AgentSnapshot(
        agent_id="initial",
        parent_id=None,
        score=float(perf.get("accuracy_score") or 0.0),
        step=0,
        resolved_ids=tuple(perf.get("total_resolved_ids") or ()),
        unresolved_ids=tuple(perf.get("total_unresolved_ids") or ()),
        empty_patch_ids=tuple(perf.get("total_emptypatch_ids") or ()),
        root=True,
    )


def candidate_dir(output_dir: str, run_id: str) -> str:
    return os.path.join(os.path.abspath(output_dir), run_id)


def submitted_instances(metadata: dict) -> int:
    perf = metadata.get("overall_performance") or {}
    deep = metadata.get("overall_performance_deep") or {}
    if not isinstance(perf, dict):
        perf = {}
    if not isinstance(deep, dict):
        deep = {}
    return int(perf.get("total_submitted_instances") or 0) + int(deep.get("total_submitted_instances") or 0)


def failure_metadata(parent_id: str, entry: str | None, exc: BaseException) -> dict:
    return {
        "run_id": f"failed-{uuid.uuid4().hex[:8]}",
        "parent_commit": parent_id,
        "entry": entry,
        "model_patch_notempty": False,
        "evopilot_structural": True,
        "evopilot_failure_cluster": "infrastructure_error",
        "error": repr(exc),
        "overall_performance": {},
    }


def call_self_improve(**kwargs):
    try:
        from self_improve_step import self_improve
    except Exception as exc:
        return failure_metadata(kwargs.get("parent_commit", "initial"), kwargs.get("entry"), exc)
    try:
        return self_improve(**kwargs)
    except Exception as exc:
        return failure_metadata(kwargs.get("parent_commit", "initial"), kwargs.get("entry"), exc)


def build_evidence(metadata: dict, controller: RecoveryController, *, debug_round: int, escalations_used: int, already_counted: int, patch_path: str | None) -> VerificationEvidence:
    assert controller.active is not None
    return evidence_from_dgm_metadata(
        metadata,
        base=controller.active,
        debug_round=debug_round,
        escalations_used=escalations_used,
        lineage_id=controller.lineage_id,
        patch_path=patch_path,
        results_dir=os.path.dirname(patch_path) if patch_path else None,
        already_counted_evaluations=already_counted,
        catastrophic_drop=controller.policy.catastrophic_drop,
    )


def reevaluate_candidate(output_dir: str, metadata: dict, task_list: list, *, polyglot: bool, num_evals: int) -> dict:
    from self_improve_step import run_harness_polyglot, run_harness_swe
    from utils.evo_utils import get_model_patch_paths

    run_id = metadata["run_id"]
    folder = candidate_dir(output_dir, run_id)
    parent = metadata.get("parent_commit") or "initial"
    patch_files = list(get_model_patch_paths(os.path.abspath("."), os.path.abspath(output_dir), parent))
    own_patch = os.path.join(folder, "model_patch.diff")
    if os.path.exists(own_patch):
        patch_files.append(own_patch)
    entry = metadata.get("entry")
    if polyglot:
        run_harness_polyglot(
            entry, run_id, patch_files, num_evals, folder, metadata, run_id, None, task_list, None,
        )
    else:
        run_harness_swe(
            entry, run_id, patch_files, num_evals, folder, metadata, run_id, None, task_list, None,
        )
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def drive_escalations(controller, evidence, rerun):
    """Turn an escalate decision into one broader sample, then a terminal decision."""

    guard = 0
    while True:
        result = controller.step(evidence)
        if result.decision != Decision.ESCALATE:
            return result, evidence
        if guard >= controller.policy.max_escalations:
            evidence = with_escalation_fallback(evidence)
            result = controller.step(evidence)
            return result, evidence
        guard += 1
        extra = rerun(evidence)
        evidence = extra if extra is not None else with_escalation_fallback(evidence)


def run_live(args: argparse.Namespace, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "evopilot_state.json")
    if args.continue_from:
        if not os.path.exists(state_path):
            raise RuntimeError(f"No evopilot_state.json in {output_dir}. Cannot resume.")
        controller = load_run(output_dir)
    else:
        prepare_initial_copy(output_dir, args.benchmark)
        controller = RecoveryController(
            policy=VerifyPolicy(
                debug_budget=args.debug_budget,
                accept_margin=args.accept_margin,
                catastrophic_drop=args.catastrophic_drop,
            ),
            bank_size=args.bank_size,
        )
        controller.bootstrap(snapshot_from_initial_metadata(os.path.join(output_dir, "initial", "metadata.json")))
        flush_run(output_dir, controller, mode="live", disclaimer=LIVE_DISCLAIMER)

    small, medium = load_task_lists(args.benchmark)
    polyglot = args.benchmark == "polyglot"
    log_path = os.path.join(output_dir, "evopilot_outer.log")

    def log(message: str) -> None:
        line = message.rstrip()
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(
        "EvoPilot live run. This calls the DGM self-improve container and the "
        f"{args.benchmark} harness. API keys and Docker are required."
    )
    while controller.generation < args.max_generation:
        controller.maybe_escape_stagnation()
        context = controller.replan_context()
        assert controller.active is not None
        entry = controller.suggest_entry()
        if entry is None and small:
            entry = small[0]
        if not entry:
            raise RuntimeError("No task entry is available on the active agent.")
        controller.begin_attempt(f"self-edit targeting {entry}", entry)
        suffix = format_improve_suffix(context)
        threshold = 0.0 if context.increase_verification else args.eval_threshold
        more = None if args.shallow_eval else medium
        log(f"generation {controller.generation} improve parent={controller.active.agent_id} entry={entry}")
        metadata = call_self_improve(
            parent_commit=controller.active.agent_id,
            output_dir=output_dir,
            entry=entry,
            num_evals=args.num_evals,
            post_improve_diagnose=args.post_improve_diagnose,
            test_task_list=small,
            test_more_threshold=None if args.shallow_eval else threshold,
            test_task_list_more=more,
            full_eval_threshold=None,
            polyglot=polyglot,
            problem_statement_suffix=suffix or None,
            evopilot_role="improve",
        )
        evidence = build_evidence(
            metadata,
            controller,
            debug_round=0,
            escalations_used=0,
            already_counted=0,
            patch_path=os.path.join(candidate_dir(output_dir, str(metadata.get("run_id"))), "model_patch.diff"),
        )

        def rerun_fixed(current: VerificationEvidence):
            if args.shallow_eval or not medium or str(current.candidate_id).startswith("failed"):
                return None
            counted = submitted_instances(metadata)
            try:
                updated = reevaluate_candidate(
                    output_dir, metadata, medium, polyglot=polyglot, num_evals=args.num_evals,
                )
            except Exception as exc:
                log(f"escalate re-evaluation failed: {exc}")
                return None
            return build_evidence(
                updated,
                controller,
                debug_round=current.debug_round,
                escalations_used=current.escalations_used + 1,
                already_counted=counted,
                patch_path=os.path.join(candidate_dir(output_dir, str(updated.get("run_id"))), "model_patch.diff"),
            )

        result, evidence = drive_escalations(controller, evidence, rerun_fixed)
        debug_guard = 0
        while result.decision == Decision.DEBUG and debug_guard < controller.policy.debug_budget:
            debug_guard += 1
            parent = evidence.candidate_id
            if not parent or str(parent).startswith("failed"):
                parent = controller.active.agent_id
            statement = format_debug_statement(evidence, controller.replan_context())
            log(f"generation {controller.generation} debug round {evidence.debug_round + 1} parent={parent}")
            metadata = call_self_improve(
                parent_commit=parent,
                output_dir=output_dir,
                entry=entry,
                num_evals=args.num_evals,
                post_improve_diagnose=False,
                test_task_list=small,
                test_more_threshold=None if args.shallow_eval else threshold,
                test_task_list_more=more,
                full_eval_threshold=None,
                polyglot=polyglot,
                problem_statement_override=statement,
                evopilot_role="debug",
            )
            evidence = build_evidence(
                metadata,
                controller,
                debug_round=result.debug_round,
                escalations_used=0,
                already_counted=0,
                patch_path=os.path.join(candidate_dir(output_dir, str(metadata.get("run_id"))), "model_patch.diff"),
            )
            result, evidence = drive_escalations(controller, evidence, rerun_fixed)
        log(
            f"generation {controller.generation} decision={result.decision.value} "
            f"reason={result.reason} active={result.active.agent_id} score={result.active.score}"
        )
        flush_run(output_dir, controller, mode="live", disclaimer=LIVE_DISCLAIMER)
    summary = controller.summary()
    summary["mode"] = "live"
    summary["disclaimer"] = LIVE_DISCLAIMER
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args)
    if args.dry_run:
        if args.continue_from:
            print("--dry_run replays a fresh synthetic trace and ignores --continue_from")
        summary = run_smoke(
            output_dir,
            max_attempts=args.max_generation,
            bank_size=args.bank_size,
            debug_budget=args.debug_budget,
        )
    else:
        summary = run_live(args, output_dir)
    print(json.dumps({
        "output_dir": output_dir,
        "mode": summary.get("mode"),
        "disclaimer": summary.get("disclaimer"),
        "best_score": summary.get("best_score"),
        "active_score": (summary.get("active") or {}).get("score"),
        "auec_best": summary.get("auec_best"),
        "auec_active": summary.get("auec_active"),
        "cumulative_evaluations": summary.get("cumulative_evaluations"),
        "cumulative_tokens": summary.get("cumulative_tokens"),
        "decisions": summary.get("decisions"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
