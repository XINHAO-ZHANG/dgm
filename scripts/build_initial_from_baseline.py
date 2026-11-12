"""Build DGM's initial-agent cache from the qwen baseline, on EvoPilot's split.

DGM starts from a cached evaluation of the unmodified agent (``initial_polyglot/``):
its score sets the first parent's selection weight, and its unresolved problems
and their logs are what the first diagnoses read. The cache in this repository
was made with claude-3-5-sonnet on the full benchmark; the aligned baseline uses
qwen and EvoPilot's split, so it needs the same baseline EvoPilot starts from.

Under the split DGM keeps its staged screen on the evaluation problems: bucket 1
is ``small.json`` and the other buckets the second stage (DGM_outer.py --split);
in EvoPilot's default split they are the rest of ``medium.json``. The
initial agent's ``overall_performance`` therefore covers bucket 1, the problems
every child is scored on, and ``overall_performance_deep`` covers all buckets.
With several baseline runs per problem, ``accuracy_score`` is passes over runs
and a problem counts as resolved when at least half of its runs passed, as in
EvoPilot's bucket baseline. One failing run of each unresolved problem is copied
into ``predictions/`` for DGM's diagnosis. Held-out problems are not read.

    python scripts/build_initial_from_baseline.py --split split.json \\
        --baseline <calibration dir> [--baseline ...] --out initial_polyglot_qwen
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

PASS = "resolved"
FAIL = ("unresolved", "empty_patch")


def count_runs(summaries: list[dict]) -> dict[str, tuple[int, int]]:
    """(passes, runs) per problem; error and missing outcomes are not runs (EvoPilot's partition.count_runs)."""

    counts: dict[str, list[int]] = {}
    for summary in summaries:
        for instance_id, row in (summary.get("per_task") or {}).items():
            entry = counts.setdefault(instance_id, [0, 0])
            for outcome in row.get("outcomes") or ():
                if outcome == PASS:
                    entry[0] += 1
                    entry[1] += 1
                elif outcome in FAIL:
                    entry[1] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def performance(problems: list[str], counts: dict[str, tuple[int, int]]) -> dict:
    """``overall_performance`` in DGM's shape, from (passes, runs) per problem."""

    resolved = [i for i in problems if counts[i][0] * 2 >= counts[i][1]]
    unresolved = [i for i in problems if i not in resolved]
    passes = sum(counts[i][0] for i in problems)
    runs = sum(counts[i][1] for i in problems)
    return {
        "accuracy_score": passes / runs if runs else 0.0,
        "total_resolved_instances": len(resolved),
        "total_submitted_instances": len(problems),
        "total_resolved_ids": resolved,
        "total_unresolved_ids": unresolved,
        "total_emptypatch_ids": [],
        "baseline_passes": passes,
        "baseline_runs": runs,
    }


def failing_runs(sources: list[str], wanted: set[str]) -> dict[str, str]:
    """The first failing run directory (sorted) per wanted problem, across ``sources``."""

    found: dict[str, str] = {}
    for source in sources:
        for run_dir in sorted(glob.glob(os.path.join(glob.escape(source), "**", "predictions", "*"), recursive=True)):
            if not os.path.isdir(run_dir):
                continue
            for instance_id in sorted(wanted - set(found)):
                record = os.path.join(run_dir, f"{instance_id}.json")
                if not os.path.isfile(record) or not os.path.isfile(os.path.join(run_dir, f"{instance_id}.md")):
                    continue
                with open(record, "r", encoding="utf-8") as handle:
                    if json.load(handle).get("eval_result") in FAIL:
                        found[instance_id] = run_dir
    return found


def build(split_path: str, sources: list[str], out: str) -> dict:
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; this cache is written once.")
    with open(split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    buckets = [list(bucket) for bucket in split["eval_buckets"]]
    problems = [i for bucket in buckets for i in bucket]
    summaries = []
    for source in sources:
        with open(os.path.join(source, "calibration_summary.json"), "r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    counts = count_runs(summaries)
    missing = [i for i in problems if counts.get(i, (0, 0))[1] == 0]
    if missing:
        raise SystemExit(f"No usable baseline run for {len(missing)} evaluation problems ({', '.join(missing[:3])}).")
    small = performance(buckets[0], counts)
    deep = performance(problems, counts)
    wanted = set(deep["total_unresolved_ids"])
    runs = failing_runs(sources, wanted)
    lacking = sorted(set(small["total_unresolved_ids"]) - set(runs))
    if lacking:
        raise SystemExit(f"No failing run with a log for {', '.join(lacking)}: DGM could not diagnose them.")
    for instance_id, run_dir in sorted(runs.items()):
        target = os.path.join(out, "predictions", f"baseline_{os.path.basename(run_dir)}")
        os.makedirs(target, exist_ok=True)
        for suffix in (".md", ".json", "_eval.md"):
            src = os.path.join(run_dir, f"{instance_id}{suffix}")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target, f"{instance_id}{suffix}"))
    with open(split_path, "rb") as handle:
        split_digest = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "run_id": "initial",
        "source": {"split": os.path.abspath(split_path), "split_sha256": split_digest, "baseline": [os.path.abspath(s) for s in sources]},
        "baseline_runs": {i: list(counts[i]) for i in problems},
        "overall_performance": small,
        "overall_performance_deep": deep,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build DGM's initial-agent cache from the qwen baseline.")
    parser.add_argument("--split", required=True, help="Split file from EvoPilot's scripts/partition_problems.py.")
    parser.add_argument("--baseline", action="append", required=True, help="Calibration dir with calibration_summary.json. Repeatable.")
    parser.add_argument("--out", default="initial_polyglot_qwen")
    args = parser.parse_args(argv)
    metadata = build(args.split, args.baseline, args.out)
    small, deep = metadata["overall_performance"], metadata["overall_performance_deep"]
    print(
        f"{args.out}: bucket 1 accuracy {small['accuracy_score']:.3f} ({small['total_resolved_instances']}/{small['total_submitted_instances']} resolved), "
        f"all buckets {deep['accuracy_score']:.3f}; logs for {len(deep['total_unresolved_ids'])} unresolved problems"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
