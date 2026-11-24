# From Acting to Learning

[![arXiv](https://img.shields.io/badge/arXiv-2611.27159-b31b1b.svg)](https://arxiv.org/abs/2611.27159)

**Experience Selection toward Continually Learning Agents**

Yaoyuan Yan, Haoxiang Jie, Zhiyou Heng, Kailin Wang, Yumiao Li, Lifeng Liu, Anwan Yan

[Paper](https://arxiv.org/abs/2611.27159) · [PDF](https://arxiv.org/pdf/2611.27159) · [中文说明](docs/README.zh-CN.md) · [Experiments](docs/EXPERIMENTS.md)

Successful interactions are a source of training data for language agents. This project studies which action records to select, how coverage constraints affect that selection, and how the resulting student performs with a fixed executor. The repository contains ALFWorld interaction, data collection, LoRA training, adaptation, and evaluation tools, together with the earlier experiments that led to the paper.

## Results

The paper compares nine selectors across two experience pools, eight initialization seeds, and two sampling seeds: 288 trained students. Each student uses 256 distinct contents and 768 training presentations. Global selectors share the same selection freedom.

| Global selection rule | Development: API pool | Development: Generic pool | Reserved: API pool | Reserved: Generic pool |
|---|---:|---:|---:|---:|
| Random | 73.19% | 67.24% | 66.65% | 60.06% |
| High NLL | 77.34% | 69.14% | 68.80% | 61.91% |
| Optimizer-aware gradient | **81.45%** | **73.19%** | **71.63%** | **69.14%** |

The reserved evaluation reuses 97 checkpoints on 128 disjoint task groups, totaling 12,416 episodes. Averaging the two pools equally, gradient selection gains **7.03 percentage points over random** and **5.03 over high NLL**. The gradient score follows LESS; the study examines its behavior under controlled selection budgets and coverage constraints.

These are ALFWorld text-interaction results on train-derived development and reserved panels. The official seen/unseen test sets were not used. The experiments evaluate experience selection followed by SFT; repeated continual-learning cycles and long-term retention remain open questions.

## Setup

Python 3.10 or later is required. CPU tests do not need a GPU, model weights, or an API key.

```bash
python3 -m venv .venv-dev
.venv-dev/bin/python -m pip install -e . numpy matplotlib pytest
PYTHONPATH=src .venv-dev/bin/python -m pytest -q
```

ALFWorld simulation and GPU inference use separate environments. The setup scripts require uv; the model environment expects a CUDA-compatible PyTorch installation on the host.

```bash
bash scripts/setup_cpu.sh
python3 scripts/download_alfworld.py
bash scripts/lh prepare
bash scripts/lh smoke
```

The smoke check replays environment actions. For model setup, inference, and training commands, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/learning_harness/` | Environment interface, model clients, collection, data export, and evaluation |
| `scripts/` | Experiment runners, analysis, plotting, and W&B tracking |
| `configs/` | Experiment configurations and pinned external sources |
| `requirements/` | Simulator and model dependencies |
| `tests/` | CPU tests for protocols, data handling, statistics, and tracking |
| `docs/` | Research overview, protocols, reports, and setup instructions |
| `runs/` | Archived configurations, per-task results, and experiment metadata |
| `third_party/` | Selected upstream prompts and reference code, with their licenses |

The repository began as a research snapshot. Dated reports describe their state at the time of writing; the [paper](https://arxiv.org/abs/2611.27159) and the result table above summarize the completed selection study. Frozen protocols and recorded source hashes are retained for historical verification. See [Snapshot contents](docs/GITHUB_SNAPSHOT.zh-CN.md) for available assets and trace restoration.

## Experiments and tracking

- [Research overview](docs/RESEARCH_OVERVIEW.zh-CN.md): study design and relation to the collection experiments.
- [Experiment index](docs/EXPERIMENTS.md): reports, protocols, and code entry points.
- [Reproducibility](docs/REPRODUCIBILITY.md): model setup, collection, SFT, and evaluation.
- [W&B tracking](docs/TRACKING.md): logging, upload, and resume behavior.

Published results can be inspected without starting a model server. Reproducing a historical run also requires its weights and local assets, which are excluded from the snapshot. Use the protocol associated with that experiment when reconstructing those assets.

## Citation

```bibtex
@article{yan2026acting,
  title={From Acting to Learning: Experience Selection toward Continually Learning Agents},
  author={Yan, Yaoyuan and Jie, Haoxiang and Heng, Zhiyou and Wang, Kailin and Li, Yumiao and Liu, Lifeng and Yan, Anwan},
  journal={arXiv preprint arXiv:2611.27159},
  year={2026},
  doi={10.48550/arXiv.2611.27159},
  url={https://arxiv.org/abs/2611.27159}
}
```

Upstream code retains its original license. This repository does not grant an additional open-source license for the project code. Correspondence: [yanyaoyuan@bgyfw.com](mailto:yanyaoyuan@bgyfw.com).
