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

OpenRouter can serve the self-improve diagnose and coding calls. Put the OpenRouter key in `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, and point the OpenAI client at OpenRouter. Default smoke models are `gpt-4o-mini` (chat completions). Anthropic stays optional for that path; non-self-improve SWE solving still uses Claude.

```bash
export OPENAI_API_KEY='...'          # or OPENROUTER_API_KEY
export OPENROUTER_API_KEY='...'      # forwarded into the self-improve container
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
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

EvoPilot is an alternate outer loop. It keeps one active agent, a small trusted checkpoint bank (default K=4), and a verify → bounded debug (default R=2) → accept or rollback → replan controller. It does not replace the DGM archive. `DGM_outer.py` is the baseline and is unchanged.

The controller records a compute curve for later AUEC-style analysis: cumulative evaluations, best accepted score, and the score of the agent currently being evolved. Token totals are stored only when a self-improve result actually reports them. A dry run does not invent either benchmark scores or paper metrics.

Smoke path, no API keys and no container:

```bash
python3 EvoPilot_outer.py --dry_run --output_dir /tmp/evopilot_smoke
python3 -m pytest tests/test_evopilot_controller.py tests/test_apptainer_compat.py
```

The smoke command replays a fixed synthetic trace and writes:

- `evopilot_trajectory.jsonl` — one row per bootstrap, verify, accept, debug, rollback, or replan, with cumulative evaluations
- `evopilot_state.json` — active agent, checkpoint bank, and evolution memory
- `evopilot_summary.json` — AUEC numbers, salvage rate, and an explicit synthetic-run disclaimer

Live Polyglot evolution uses the same self-improve container and Polyglot harness as DGM. It needs an OpenAI-compatible key (`OPENAI_API_KEY` or `OPENROUTER_API_KEY`), `pip install -r requirements.txt`, and `python -m polyglot.prepare_polyglot_dataset`. With `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, diagnose and self-improve use `gpt-4o-mini` and Anthropic is optional. The container runtime follows `DGM_CONTAINER_RUNTIME` (`auto`, `docker`, or `apptainer`). On bach4, select Apptainer. A Docker daemon is not required for that path. Image build and fakeroot notes are in [APPTAINER.md](APPTAINER.md). Start with one generation; each generation is a full self-modify plus evaluation.

```bash
export PATH="$HOME/software/apptainer/bin:$PATH"
export DGM_CONTAINER_RUNTIME=apptainer
python3 EvoPilot_outer.py --benchmark polyglot --max_generation 1 --output_dir output_evopilot/polyglot_smoke
```

SWE-bench uses `--benchmark swe`. Accept and rollback compare accuracy on one fixed problem set: `small.json` with `--shallow_eval`, otherwise the union of `small.json` and `medium.json`. The cached full-suite accuracy is rescaled onto that set from per-problem results; it is not compared directly with a subset score. `--eval_threshold` does not switch the set. Resume with `--continue_from output_evopilot/<run_id>` after `evopilot_state.json` exists.

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
- `evopilot/` controller state, verify policy, checkpoint bank, evolution memory, and AUEC ledger

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
