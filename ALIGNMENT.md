# DGM baseline aligned with EvoPilot

EvoPilot (`../EvoPilot`) is compared with DGM. For the comparison to measure the
methods and nothing else, DGM here runs the same code, the same model, and the same
problems as EvoPilot, and keeps its own method.

## What is shared

`scripts/sync_runtime.py` copies these from an EvoPilot checkout in which they are
committed. `EVOPILOT_RUNTIME.json` records that EvoPilot commit and a hash per file.

| Group | Files | Why it must match |
| --- | --- | --- |
| Coding agent | `coding_agent_polyglot.py`, `llm.py`, `llm_withtools.py`, `tools/`, `prompts/tooluse_prompt.py`, `prompts/testrepo_prompt.py` | The agent both methods start from and every task is solved with. EvoPilot's `llm_withtools.py` recovers unambiguous tool-call slips; without it the starting agent is weaker. |
| Packages | `requirements.txt` | Installed in the self-improve image and in every task sandbox. |
| Grading and task images | `polyglot/harness.py`, `constants.py`, `dockerfiles.py`, and the other harness modules | EvoPilot runs every Java test (not only the first), bakes Gradle and the Rust/Java toolchains into the images, and refuses a missing toolchain instead of calling it unresolved. |
| Sandbox runtime | `utils/apptainer_compat.py`, `docker_utils.py`, `host_resources.py`, ... | Same container behaviour and resource gate. |
| Self-improve image | `Dockerfile`, `.dockerignore` | What the self-editing agent sees of its repository: only the agent (20 files). Controllers, tests, scripts, and docs stay on the host in both repositories. |

The dataset (`polyglot/polyglot_benchmark_metadata.json`) is compared, not copied.
The model is `qwen/qwen3-coder-flash` in both (`llm.py`).

## What stays DGM's

- `DGM_outer.py`: the archive, parent selection by score and child count, `keep_all`.
- `self_improve_step.py`: DGM's Improve and its staged screen.
- `prompts/self_improvement_prompt.py`: the upstream diagnosis prompt. EvoPilot's
  diagnosis also states a target failure, expected impact, and regression risk
  (its design doc §5.1); DGM-Original does not.
- The self-improve image is tagged `dgm-baseline` (`SELF_IMPROVE_IMAGE`), not `dgm`:
  images are cached by tag in a shared directory, and EvoPilot builds `dgm` from its
  own checkout.

## Problems

Both methods evolve on DGM's own screen, the problems DGM and HGM report on.
EvoPilot's default split (`scripts/partition_problems.py`, layout `screen`) keeps it as
it is, and DGM reads it with `--split`:

- Bucket 1 is `small.json` (10 problems); every child is scored on it, and parent
  selection and the diagnosis problem use that score and its unresolved problems,
  as upstream.
- A child that resolves at least 40% of bucket 1 also runs the other buckets, the
  rest of `medium.json` (49 problems; the one `medium.json` problem already in
  `small.json` is not run twice); that result is `overall_performance_deep`, as
  upstream.
- The held-out set (40 problems by default) comes from outside the screen. Neither
  method runs or reads it while evolving; both report on it at the end.

EvoPilot measures the same 59 problems differently (a paired comparison with its
active agent, one bucket at a time) and diagnoses one failing training problem, as
DGM does; its diagnosis also sees a short history of earlier edits.

The initial agent's cache comes from the same qwen baseline as EvoPilot's
(`scripts/build_initial_from_baseline.py`), not from `initial_polyglot/`, which was
made with claude-3-5-sonnet on the full benchmark.

## Running it

```bash
python scripts/sync_runtime.py                 # from ../EvoPilot, committed files only
python scripts/sync_runtime.py --check         # record, and the self-improve file sets
python scripts/build_initial_from_baseline.py --split <split.json> \
    --baseline <calibration dir> [--baseline ...] --out initial_polyglot_qwen
python DGM_outer.py --polyglot --split <split.json> --initial_dir initial_polyglot_qwen
```

`DGM_outer.py --split` refuses to start while a runtime file differs from the record,
the record was made from uncommitted EvoPilot files, or the initial cache was built
from another split. Set the Apptainer environment as in EvoPilot's `CLAUDE.md`
(`APPTAINER_IMAGE_DIR` in particular), and give both methods the same resource flags
(`--workspace-root`, `--unpack-reserve-gb`, see EvoPilot's `APPTAINER.md`). Runs of
the two on one workspace root share its unpack reservations.

## Known differences

- `.gitignore` and `.gitattributes` are each repository's own. Inside the
  self-improve container they decide which files the agent creates are tracked
  (EvoPilot also ignores `.env`) and normalize line endings (EvoPilot: `eol=lf`).
- The image carries each repository's `.git`, so `git log` there shows a different
  history.
- Compute is counted from each child's `overall_performance(_deep)`; DGM writes no
  ledger of its own.
