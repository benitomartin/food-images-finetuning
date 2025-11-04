# FOOD IMAGES FINE TUNING

![Diagram](static/image.png)

<div align="center">

<!-- Project Status -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<!-- Providers -->

</div>

## Table of Contents

- [Food Images Fine Tuning](#food-images-fine-tuning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Evaluation Results](#evaluation-results)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Training on Modal](#training-on-modal)
    - [Evaluation](#evaluation)
    - [Testing](#testing)
    - [Quality Checks](#quality-checks)
  - [License](#license)

## Overview

Fine-tuning of LiquidAI LFM2-VL vision-language models on food image classification using LoRA.
The project supports training on Modal cloud infrastructure and includes evaluation tools for testing model performance.

## Evaluation Results

### 3 Classes Evaluation

Comparison of model performance on 3-class food image classification:

| Metric | 450M Base | 450M Fine-tuned | 1.6B Base |
|--------|----------|----------------|-----------|
| **Overall Accuracy** | 89.1% (401/450) | 94.7% (426/450) | **96.4%** (434/450) |
| hamburger | 94.6% (139/147) | 96.6% (142/147) | **98.0%** (144/147) |
| garlic_bread | 79.0% (128/162) | **95.1%** (154/162) | 93.2% (151/162) |
| hot_dog | 95.0% (134/141) | 92.2% (130/141) | **98.6%** (139/141) |

**Summary:** Fine-tuning the 450M model improves accuracy from 89.1% to 94.7% (+5.6 percentage points), bringing it close to the 1.6B base model performance (96.4%).

### 5 Classes Evaluation

Comparison of model performance on 5-class food image classification:

| Metric | 450M Base | 450M Fine-tuned | 1.6B Base |
|--------|----------|----------------|-----------|
| **Overall Accuracy** | 87.1% (653/750) | 91.5% (686/750) | **95.1%** (713/750) |
| hamburger | 86.1% (130/151) | **96.0%** (145/151) | 95.4% (144/151) |
| garlic_bread | 84.5% (125/148) | 87.2% (129/148) | **95.3%** (141/148) |
| hot_dog | 97.1% (133/137) | 96.4% (132/137) | **99.3%** (136/137) |
| ceviche | 74.0% (125/169) | 81.7% (138/169) | **92.3%** (156/169) |
| carrot_cake | 96.6% (140/145) | **97.9%** (142/145) | 93.8% (136/145) |

**Summary:** Fine-tuning the 450M model improves accuracy from 87.1% to 91.5% (+4.4 percentage points) on the 5-class task.

## Project Structure

```text
├── src/
│   └── food_images_finetuning/
│       ├── configs/          # Model and fine-tuning configuration files
│       ├── evaluation/       # Model evaluation scripts
│       ├── loaders/          # Dataset and model loaders
│       ├── training/         # Modal cloud training scripts
│       └── schemas/          # Pydantic models for validation
├── config.py                 # Settings
├── models/                   # Saved model outputs
├── notebooks/                # Jupyter notebooks
├── Makefile                  # Development commands
├── pyproject.toml            # Project dependencies and config
└── README.md                 # README
```

## Prerequisites

- Python 3.12
- uv package manager
- Modal account (for cloud training)
- Hugging Face account

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd product-images-finetuning
   ```

2. Create a virtual environment:

   ```bash
   uv venv
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Install the required packages:

   ```bash
   uv sync --all-groups --all-extra
   ```

5. Set up Modal (for cloud training):

   ```bash
   modal token new
   ```

6. Set up Hugging Face Token:

   ```bash
   cp .env.example .env
   ```

## Usage

### Configuration

Training is configured via YAML files in `src/food_images_finetuning/configs/`:

- **Model configs** (`config_*.yaml`): Specify base model and dataset
- **Fine-tuning configs** (`finetune_*.yaml`): Configure LoRA, training parameters, and runtime settings

Key configuration sections:

- `lora`: LoRA adapter configuration (rank, alpha, dropout)
- `sft`: Supervised fine-tuning parameters (learning rate, epochs, batch size)
- `runtime`: Modal runtime settings (GPU, timeout, volume name)
- `processor`: Model processor settings
- `train_data`: Dataset configuration (classes, samples per class, test split)
- `prompts`: System and user messages for training

### Training on Modal

Train a model using Modal cloud infrastructure:

```bash
modal run src/food_images_finetuning/training/train_on_modal.py \
  --finetune-config src/food_images_finetuning/configs/finetune_0.yaml \
  --config src/food_images_finetuning/configs/config_450M.yaml
```

### Evaluation

Evaluate a fine-tuned model:

```bash
uv run src/food_images_finetuning/evaluation/evaluate_fine_tune_model.py \
  --config src/food_images_finetuning/configs/config_450M.yaml \
  --finetune-config src/food_images_finetuning/configs/finetune_0.yaml
```

Evaluate the base model (before fine-tuning):

```bash
uv run src/food_images_finetuning/evaluation/evaluate_base_model.py \
  --config src/food_images_finetuning/configs/config_450M.yaml \
  --finetune-config src/food_images_finetuning/configs/finetune_0.yaml
```

### Quality Checks

Run all quality checks (lint, format, type check, clean):

```bash
make all
```

Individual Commands:

- Display all available commands:

  ```bash
  make help
  ```

- Check code formatting and linting:

  ```bash
  make all-check
  ```

- Fix code formatting and linting:

  ```bash
  make all-fix
  ```

- Clean cache and build files:

  ```bash
  make clean
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
