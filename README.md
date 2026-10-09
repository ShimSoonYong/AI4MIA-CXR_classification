# AI4MIA COVID-19 Image Classification Project

**Author:** Shim Soonyong  
**Date:** 2026/10/01  

## Overview

This project aims to develop and evaluate **Convolutional Neural Network (CNN)** models for classifying COVID-19 status using chest X-ray images.

As shown in the directory structure under `results/`, this project focuses on the following key comparative experiments:
1. **Comparison between scratch and pretrained models**: Evaluating performance differences when training models from scratch versus leveraging pretrained feature extractors (e.g., AlexNet, GoogleNet, ResNet).
2. **Comparison between raw data and augmented data**: Analyzing the impact of data augmentation (`_all` vs `_none`) on model generalizability and clinical metrics.

## Project Structure

The project directory is structured as follows to ensure reproducibility and modularity:

```text
assignment1/
├── .devcontainer/              # Dev container configurations for consistent environments
├── dataset/                    # Dataset directory (excluded via .gitignore)
├── figures/                    # Generated project-wide visualization figures
├── notebooks/                  # Jupyter Notebooks for EDA, training, and evaluation
├── results/                    # Experiment results categorized by model, pretraining, and augmentation
│   ├── alexnet_pretrained_all/ # AlexNet, Pretrained, Augmented data
│   ├── alexnet_pretrained_none/# AlexNet, Pretrained, Raw data (no augmentation)
│   ├── alexnet_scratch_all/    # AlexNet, Scratch, Augmented data
│   ├── alexnet_scratch_none/   # AlexNet, Scratch, Raw data (no augmentation)
│   ├── googlenet_.../          # GoogleNet variations
│   ├── resnet_.../             # ResNet variations
│   └── metrics.json            # Final performance metrics record across models
├── src/                        # Reusable source code files (.py)
│   ├── __init__.py
│   ├── dataset.py              # Custom Dataset and DataLoader definitions
│   ├── main.py                 # Main execution pipeline and argument parser
│   ├── models.py               # Model architectures (Scratch CNN and Transfer Learning models)
│   ├── train.py                # Training loops and Trainer class definition
│   └── utils.py                # Metric calculation utilities (Sensitivity, Specificity, AUROC)
├── AI4MIA_Assignment1_Dataset.zip # Compressed raw dataset archive (excluded via .gitignore)
├── Dockerfile                  # Explicitly exposed Dockerfile for environment reproduction
├── .gitignore                  # Git untracked files configuration
├── README.md                   # Project documentation

```

## Getting Started

### 1. Environment Setup

You can set up the environment using either Docker or VS Code Dev Containers based on the root `Dockerfile`.
Alternatively, install the required python packages manually:

```bash
pip install -r requirements.txt

```

### 2. Dataset Preparation

The dataset archive (`AI4MIA_Assignment1_Dataset.zip`) is excluded from GitHub due to repository size limitations and course security policies. However, you can set up your own dataset by arranging your chest X-ray images inside the `dataset/` directory.

The image dataset must follow the standard PyTorch `ImageFolder` directory structure as shown below:

```text
dataset/
├── covid/
│   ├── image1.png
│   └── image2.png
├── normal/
│   └── ...
└── viral_pneumonia/
    └── ...

```


### 3. Model Training & Evaluation

1. **Exploratory Data Analysis (EDA)**: Run the EDA using the Jupyter notebooks provided in the `notebooks/` directory.
2. **Main Training & Evaluation Pipeline**: Execute the training and evaluation pipeline via `src/main.py` using terminal command-line arguments.

#### **Command-Line Arguments Reference**

| Category | Argument | Type | Default | Description |
| --- | --- | --- | --- | --- |
| **Directory Configs** | `--raw_dir` | `str` | `'dataset/raw'` | Path to the raw dataset directory |
|  | `--processed_dir` | `str` | `'dataset/processed'` | Path to the preprocessed dataset directory |
|  | `--results_dir` | `str` | `'results'` | Root directory where experiment results and logs are saved |
| **Data Loading** | `--use_saved` | Flag | `False` | Use preprocessed `.pt` tensor datasets if enabled |
|  | `--save_processed` | Flag | `False` | Convert and save raw data into `.pt` tensors |
|  | `--num_workers` | `int` | `4` | Number of subprocesses to use for data loading |
|  | `--augment` | `str` | `'none'` | Data augmentation strategy (`none`, `all`, `flip`, `rotation`, `crop`) |
| **Model & Hyperparams** | `--model_name` | `str` | `'resnet'` | Target model architecture (`resnet`, `alexnet`, `googlenet`, `all`) |
|  | `--scratch` | Flag | `False` | Train from scratch without loading pretrained weights |
|  | `--epochs` | `int` | `20` | Maximum number of training epochs |
|  | `--batch_size` | `int` | `32` | Number of samples per training batch |
|  | `--lr` | `float` | `0.001` | Initial learning rate for optimization |
|  | `--seed` | `int` | `42` | Random seed for full reproducibility |

#### **Example Command**

To train a ResNet model from scratch with full data augmentation (`all`) for 20 epochs, run:

```bash
python src/main.py --model_name resnet --scratch --augment all --epochs 20

```

Evaluation results, model checkpoints, and metrics will be automatically saved under the `results/` folder, categorized by each experimental setup you choose.