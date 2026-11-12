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

## Baseline for EvoPilot

On the `dgm-baseline-alignment` branch, DGM is the baseline that EvoPilot (`../EvoPilot`) is compared with. It runs EvoPilot's coding agent, harness, sandbox code, model (`qwen/qwen3-coder-flash` through OpenRouter), and problems, and keeps DGM's own method. [ALIGNMENT.md](ALIGNMENT.md) lists what is shared, what stays DGM's, and how to run it. Shared files are copied from EvoPilot with `scripts/sync_runtime.py`; change them there, not here.

The container runtime follows `DGM_CONTAINER_RUNTIME` (`auto`, `docker`, or `apptainer`). The LIG servers have no Docker daemon; select Apptainer. Image build and fakeroot notes are in [APPTAINER.md](APPTAINER.md).

Before each Improve call and before Verify starts another sandbox, the run checks free space on `/tmp` (and `TMPDIR` / `APPTAINER_TMPDIR` / `APPTAINER_WORKSPACE_ROOT` when those paths differ) plus MemAvailable and memory pressure. Defaults are `--min-tmp-free-gb 10`, `--min-available-mem-gb 4`, and `--max-mem-pressure 10` (PSI full avg10, percent). A value of `0` disables the disk or memory floor; a negative `--max-mem-pressure` disables the PSI check. If a check fails, that generation does not call the model and does not start a sandbox. Each sandbox unpack also holds `--unpack-reserve-gb` (default 9) until it lands, and `--workspace-root` puts sandboxes elsewhere, for example in `/dev/shm`.

`/tmp` filling up, an out-of-memory error, an LDAP/userid lookup failure, or an Apptainer start failure sets `infra_error_code` and stops the outer loop. `--continue-on-infra-error` goes on to the next generation anyway. It does not bypass the disk and memory gates. After each generation the run deletes only Apptainer workspace directories stamped with this output directory's owner. Other users' sandboxes are left in place. `--dry-resource-check` prints the snapshot and exits. See [APPTAINER.md](APPTAINER.md).

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
- `ALIGNMENT.md`, `EVOPILOT_RUNTIME.json`, `scripts/sync_runtime.py` alignment with EvoPilot (shared runtime and its record)

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
