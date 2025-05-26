# LiEr: Human-in-the-Loop Path-Based Link Prediction

**Status:** Experimental research code (not production-ready)

---

## Overview

LiEr (Learning interactively by Explanations to Reason) is a path-based link prediction framework that integrates human feedback to align its reasoning with valid, human-understandable inference patterns over knowledge graphs.

Key features:

* **Human-in-the-loop** reward shaping via preference-based feedback
* **Reinforcement learning** agent that walks knowledge graphs to predict missing links
* **Interpretability**: generated paths serve as transparent reasoning traces

LiEr matches state-of-the-art performance on standard benchmarks and demonstrates improved generalization in datasets with high spurious correlation (`Clever Hans` phenomena).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Running Experiments](#running-experiments)
5. [Project Structure](#project-structure)
6. [Contributing](#contributing)
7. [Citation](#citation)
8. [License](#license)

---

## Prerequisites

* **Python:** 3.11 (tested)
* **CUDA:** 11.2 (tested, for GPU training)
* **OS:** Linux / macOS

> **Note:** This repository contains highly experimental research code. It is provided for reproducibility and extension of our work; it is **not** designed for production use.

---

## Installation


   ```
   Install the package in development mode and required dependencies in your virtual environment:

   ```bash
   python setup.py develop
   pip install -r requirements.txt
   ```

---

## Usage

Experiments are run via the `run_training.sh` wrapper.

```bash

bash run_training.sh 
```

**run\_training.sh**: orchestrates environment variables, logging, and job submission.

Results will be saved under:

```
results/<experiment_prefix>
```

Each subfolder contains logs, and evaluation metrics.

---


## Project Structure

```
├── config/                       # Experiment configuration files
├── misc/                         # Utilities (HPO script, Optuna dashboard)
├── results/                      # Output directory for logs and metrics
├── src/
│   ├── interactive_reward_modules/  # Human-feedback collectors & reward estimators
│   │   ├── feedback.py
│   │   ├── pairs_storage.py
│   │   └── preference_based_reward.py
│   ├── knowledge_graphs/         # KG loading and preprocessing utilities
│   ├── layers/                   # Custom [LSTM layers](https://github.com/seba-1511/lstms.pth)
│   ├── models/                   # Stored policy networks
│   ├── baseline.py
│   ├── embedding.py
│   ├── environment.py
│   ├── evaluation.py
│   ├── logger.py
│   ├── policy_network.py
│   ├── reward_estimator.py
│   ├── training.py
│   └── utility.py
├── main.py                       # Entry point
├── run_training.sh               # Experiment driver script
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installer
└── README.md                     # This file
```

---

## Contributing

Contributions are welcome! To propose changes:

1. Fork the repository and create a feature branch.
2. Submit a pull request with a clear description of your changes.

Please note that this codebase is under active development; APIs and file formats may change.

---

## Citation

If you use LiEr in your research, please cite our paper:

```bibtex
@article{wehner2025lier,
  title   = {Aligning Path-based Link Prediction with Human Understanding of Valid Reasoning},
  author  = {Christoph Wehner},
  journal = {},
  year    = {2025},
}
```

---

## License

This project is released under the MIT License.
