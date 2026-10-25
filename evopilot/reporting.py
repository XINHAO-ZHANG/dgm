"""Persist the controller ledger and a resume snapshot.

``evopilot_trajectory.jsonl`` is the compute curve. ``evopilot_state.json``
is the bank, memory, and active agent. ``evopilot_summary.json`` repeats the
AUEC numbers with an explicit disclaimer so a dry run cannot be mistaken
for a benchmark result.
"""

from __future__ import annotations

import json
import os

from evopilot.controller import RecoveryController


def _write(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def flush_run(output_dir: str, controller: RecoveryController, *, mode: str, disclaimer: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    trajectory = os.path.join(output_dir, "evopilot_trajectory.jsonl")
    new_events = controller.ledger.events[controller.flushed_events :]
    with open(trajectory, "a", encoding="utf-8") as handle:
        for event in new_events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    controller.flushed_events = len(controller.ledger.events)

    state = controller.to_dict()
    state["flushed_events"] = controller.flushed_events
    _write(os.path.join(output_dir, "evopilot_state.json"), state)

    summary = controller.summary()
    summary["mode"] = mode
    summary["disclaimer"] = disclaimer
    _write(os.path.join(output_dir, "evopilot_summary.json"), summary)
    return summary


def load_trajectory(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, "evopilot_trajectory.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_run(output_dir: str) -> RecoveryController:
    state_path = os.path.join(output_dir, "evopilot_state.json")
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    controller = RecoveryController.from_dict(state)
    events = load_trajectory(output_dir)
    controller.ledger.events = events
    if events:
        controller.ledger.cumulative_evaluations = int(events[-1]["cumulative_evaluations"])
        controller.ledger.cumulative_tokens = events[-1].get("cumulative_tokens")
    controller.flushed_events = len(events)
    return controller
