<h1 align="center">
    Darwin Gödel Machine:<br/>Open-Ended Evolution of Self-Improving Agents
</h1>

<p align="center">
  <a href="https://github.com/jennyzzt/dgm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2505.22954"><img src="https://img.shields.io/badge/arXiv-2505.22954-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://sakana.ai/dgm/"><img src="https://img.shields.io/badge/-Blog-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://x.com/SakanaAILabs/status/1928272612431646943"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
  <a href="https://drive.google.com/drive/folders/1Kcu9TbIa9Z50pJ7S6hH9omzzD1pxIYZC?usp=sharing"><img src="https://img.shields.io/badge/Experiment%20Logs-4285F4?style=for-the-badge&logo=googledrive&logoColor=white"></a>
</p>


Repository for **Darwin Gödel Machine (DGM)**, a novel self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks.

<p align="center">
  <img src="./misc/overview.gif" width="100%" height="auto" />
</p>
<!-- <p align="center">
<img src="./misc/conceptual.svg"/></a><br>
</p> -->


## Setup
```bash
# API keys, add to ~/.bashrc
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

OpenRouter serves self-improve and Verify (the coding agent that solves benchmark tasks). Put the OpenRouter key in `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, and point the OpenAI client at OpenRouter. The Scalability MVP backbone is `qwen/qwen3-coder-flash` (chat completions) for both paths. `DGM_EVAL_MODEL` overrides Verify only. A `bedrock/…` id still uses Amazon Bedrock when `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION_NAME` are set. Polyglot and SWE harnesses forward these variables into the Apptainer sandbox (`llm_container_env`); the benchmark SIF does not need a rebuild for that.

```bash
export OPENAI_API_KEY='...'          # or OPENROUTER_API_KEY
export OPENROUTER_API_KEY='...'      # forwarded into self-improve and Verify sandboxes
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
# optional Verify override (default is qwen/qwen3-coder-flash):
# export DGM_EVAL_MODEL='qwen/qwen3-coder-flash'
```

```bash
# Verify that Docker is properly configured in your environment.
docker run hello-world
 
# If a permission error occurs, add the user to the Docker group
sudo usermod -aG docker $USER
newgrp docker
```

HPC login nodes that have Apptainer (or Singularity) and no Docker daemon use the same evaluation entry points. `DGM_CONTAINER_RUNTIME=auto` selects Apptainer when `docker info` fails. See [APPTAINER.md](APPTAINER.md) for a user-local install such as `~/software/apptainer` on bach4.

```bash
export PATH="$HOME/software/apptainer/bin:$PATH"
export DGM_CONTAINER_RUNTIME=apptainer
python scripts/container_smoke.py
```

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: for running analysis
sudo apt-get install graphviz graphviz-dev
pip install -r requirements_dev.txt
```

```bash
# Clone SWE-bench
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36
pip install -e .
cd ../../

# Prepare Polyglot
# Make sure git is properly configured in your environment with username and email
python -m polyglot.prepare_polyglot_dataset
```

## Running the DGM
```bash
python DGM_outer.py
```
By default, outputs will be saved in the `output_dgm/` directory.

Polyglot uses the same entry point:

```bash
python DGM_outer.py --polyglot
```

## EvoPilot recovery controller

EvoPilot is an alternate outer loop. It keeps one active agent, a small trusted checkpoint bank (default K=4), and a verify → bounded debug (default R=2) → accept, rollback, or checkmate → replan controller. It does not replace the DGM archive. `DGM_outer.py` remains the archive baseline. Both outer loops share a host resource gate so a full disk, memory pressure, or a broken Apptainer start stops the run instead of starting another generation.

After Verify, the default decider is an LLM Control Agent on the same backbone as Improve (`qwen/qwen3-coder-flash` via OpenRouter). It reads one evidence pack and returns `accept`, `debug`, `rollback`, or `checkmate`. The outer loop executes that action and does not replace it with a score threshold. `--controller policy` restores the rule-based `VerifyPolicy`. `--control_model` overrides the agent model. `--dry_run` stays on the synthetic policy trace and does not call a model. The pack is built from the candidate directory: whether `model_patch.diff` is empty and which files it changes; Improve/Debug logs (`self_evo.md`, and `self_improve.log` when that is where the tool trace landed) for tool-parse failures and view calls versus writes; Verify score, resolved / unresolved / empty-patch ids, compile and structure errors, and whether the protocol is comparable; the checkpoint bank and the active agent; and this lineage's debug count plus recent failure intents.

The controller records a compute curve for later AUEC-style analysis: cumulative evaluations, best accepted score, and the score of the agent currently being evolved. Token totals are stored only when a self-improve result actually reports them. A dry run does not invent either benchmark scores or paper metrics.

Smoke path, no API keys and no container:

```bash
python3 EvoPilot_outer.py --dry_run --output_dir /tmp/evopilot_smoke
python3 EvoPilot_outer.py --dry-resource-check
python3 -m pytest tests/test_evopilot_controller.py tests/test_control_agent.py tests/test_apptainer_compat.py tests/test_host_resources.py
```

The smoke command replays a fixed synthetic trace and writes:

- `evopilot_trajectory.jsonl` — one row per bootstrap, verify, accept, debug, rollback, or replan, with cumulative evaluations
- `evopilot_state.json` — active agent, checkpoint bank, and evolution memory
- `evopilot_summary.json` — AUEC numbers, salvage rate, and an explicit synthetic-run disclaimer

Live Polyglot evolution uses the same self-improve container and Polyglot harness as DGM. It needs an OpenAI-compatible key (`OPENAI_API_KEY` or `OPENROUTER_API_KEY`), `pip install -r requirements.txt`, and `python -m polyglot.prepare_polyglot_dataset`. With `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, diagnose, self-improve, and Verify use `qwen/qwen3-coder-flash`. Anthropic and Bedrock are optional. The harness copies `coding_agent_polyglot.py`, `llm.py`, and `llm_withtools.py` into the sandbox on each task, so those files do not require a `pb.*` SIF rebuild. The self-improve `dgm` image is baked with `COPY . .`; rebuild that SIF (`--force_rebuild`) only to pick up agent changes inside Improve. The container runtime follows `DGM_CONTAINER_RUNTIME` (`auto`, `docker`, or `apptainer`). On bach4, select Apptainer. A Docker daemon is not required for that path. Image build and fakeroot notes are in [APPTAINER.md](APPTAINER.md).

Polyglot Verify uses DGM's staged meter by default. No extra staging flag is required. The screen is `polyglot/subsets/small.json` (10). When `total_resolved_instances >= --eval_threshold * 10` (default `0.4`, so 4 resolved), the same candidate is also evaluated on `polyglot/subsets/medium.json` (50). That second stage is `overall_performance_deep` and counts toward cumulative task-evals. `--shallow_eval` is smoke only: it stays on the 10-problem screen and does not run medium. Accept and rollback use the screen, because every candidate finishes it; a small-set score is not compared with a medium-set or full-suite score. The cached full-suite accuracy is rescaled onto the screen from per-problem results.

```bash
export PATH="$HOME/software/apptainer/bin:$PATH"
export DGM_CONTAINER_RUNTIME=apptainer
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
# Scalability MVP. Do not pass --shallow_eval.
python3 EvoPilot_outer.py --benchmark polyglot --max_generation 1 --output_dir output_evopilot/polyglot_smoke
# Smoke only:
python3 EvoPilot_outer.py --benchmark polyglot --shallow_eval --max_generation 1 --output_dir output_evopilot/polyglot_shallow
```

SWE-bench uses `--benchmark swe` and the same staging switch. Resume with `--continue_from output_evopilot/<run_id>` after `evopilot_state.json` exists.

Before each Improve call and before Verify starts another sandbox, the run checks free space on `/tmp` (and `TMPDIR` / `APPTAINER_TMPDIR` / `APPTAINER_WORKSPACE_ROOT` when those paths differ) plus MemAvailable and memory pressure. Defaults are `--min-tmp-free-gb 10`, `--min-available-mem-gb 4`, and `--max-mem-pressure 10` (PSI full avg10, percent). A value of `0` disables the disk or memory floor; a negative `--max-mem-pressure` disables the PSI check. If a check fails, that generation does not call the model and does not start a sandbox.

`/tmp` filling up, an out-of-memory error, an LDAP/userid lookup failure, or an Apptainer start failure sets `infra_error_code` and stops the outer loop. `--continue-on-infra-error` goes on to the next generation anyway. It does not bypass the disk and memory gates. After each generation the run deletes only Apptainer workspace directories stamped with this output directory's owner. Other users' sandboxes are left in place. `--dry-resource-check` prints the snapshot and exits. The same flags exist on `DGM_outer.py`. See [APPTAINER.md](APPTAINER.md).

What is still scaffolding, not a paper run: there is no neutralized-initialization ablation, no learned recoverability model, and no claim about Polyglot or SWE-bench scores. Provider token usage is not summed inside DGM's self-improve step, so `cumulative_tokens` stays null unless that field is present on a result. Escalate uses the spread across repeated harness reports when two or more exist for the same candidate; one report is not enough to estimate variance.

## File Structure
- `analysis/` scripts used for plotting and analysis
- `initial/` SWE-bench logs and performance of the initial agent
- `initial_polyglot/` Polyglot logs and performance of the initial agent
- `swe_bench/` code needed for SWE-bench evaluation
- `polyglot/` code needed for Polyglot evaluation
- `prompts/` prompts used for foundation models
- `tests/` tests for the DGM system
- `tools/` tools available to the foundation models
- `coding_agent.py` main implementation of the initial coding agent
- `DGM_outer.py` entry point for running the DGM algorithm
- `EvoPilot_outer.py` entry point for the EvoPilot recovery controller
- `evopilot/` controller state, Control Agent, verify-policy fallback, checkpoint bank, evolution memory, and AUEC ledger

## Logs from Experiments
This [google drive folder](https://drive.google.com/drive/folders/1Kcu9TbIa9Z50pJ7S6hH9omzzD1pxIYZC?usp=sharing) contains all the foundation model output logs from the experiments shown in the paper.

## Safety Consideration
> [!WARNING]  
> This repository involves executing untrusted, model-generated code. We strongly advise users to be aware of the associated safety risks. While it is highly unlikely that such code will perform overtly malicious actions under our current settings and with the models we use, it may still behave destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.

## Acknowledgement

The evaluation framework implementations are based on the [SWE-bench](https://github.com/swe-bench/SWE-bench) and [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) repositories.

## Citing
If you find this project useful, please consider citing:
```bibtex
@article{zhang2025darwin,
  title={Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents},
  author={Zhang, Jenny and Hu, Shengran and Lu, Cong and Lange, Robert and Clune, Jeff},
  journal={arXiv preprint arXiv:2505.22954},
  year={2025}
}
```
