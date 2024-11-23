# Chest_Xray_Computer_Vision
Final Project for INFO-B627: Advanced Seminar 1

# Triplet Learning for Image Classification

This project implements a triplet learning architecture for image classification using various backbone neural networks. The system uses semi-hard triplet loss to learn discriminative embeddings that can be used for both closed-set and open-set image recognition tasks.

## Features

- Support for multiple backbone architectures:
  - ResNet (50, 101, 152, and V2 variants)
  - MobileNetV2
  - Xception
  - VGG19
  - DenseNet201
  - InceptionV3
  - InceptionResNetV2
- Flexible dataset handling with automatic train/validation/test splitting
- Supports both open-set and closed-set evaluation
- Comprehensive data augmentation pipeline
- GPU acceleration support
- Mean Average Precision (mAP) evaluation metrics
- Experiment tracking and result logging

## Requirements

- Python 3.11+
- TensorFlow
- TensorFlow Addons
- CuPy (for GPU acceleration)
- NumPy
- CUDA Toolkit 12.2 (for GPU support)

## Project Structure

```
.
├── dataset.py            # Dataset loading and preprocessing utilities
├── experiment.py         # Main training and evaluation pipeline
├── mean_average_precision.py  # Evaluation metrics implementation
└── inceptionv3.sh       # Example SLURM batch script for training
```

## Installation

1. Set up a conda environment:
```bash
conda create -n triplet python=3.11.5
conda activate triplet
```

2. Install required modules:
```bash
module load python/gpu/3.11.5
module load cudatoolkit/12.2
```

## Usage

### Basic Training

To train a model, use the `experiment.py` script with appropriate parameters:

```bash
python experiment.py \
    --backbone "InceptionV3" \
    --dataset="/path/to/dataset" \
    --output="output.zip" \
    --batch_size 32 \
    --embedding_size 128 \
    --retrain_layer_count 223 \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --augmentation_count 4 \
    --augmentation_factor 0.2 \
    --loss_margin 0.5 \
    --train_epochs 50
```

### Parameters

#### Required Parameters:
- `--backbone`: The backbone architecture to use (e.g., "ResNet50", "InceptionV3")
- `--dataset`: Path to the image dataset
- `--output`: Path where experiment results will be saved (must end in .zip)

#### Optional Parameters:
- `--batch_size`: Number of samples per batch (default: 32)
- `--learning_rate`: Learning rate for training (default: 0.001)
- `--dropout`: Dropout rate for regularization (default: 0.1)
- `--augmentation_count`: Number of augmentations per image (default: 4)
- `--augmentation_factor`: Intensity of augmentations (default: 0.1)
- `--loss_margin`: Margin for triplet loss (default: 0.5)
- `--embedding_size`: Output embedding dimensions (default: 128)
- `--retrain_layer_count`: Number of layers to retrain from base model
- `--train_epochs`: Total training epochs (default: 50)
- `--save_model`: Whether to save the trained model (default: False)
- `--vote_count`: Number of votes per embedding in closed eval mode (default: 5)
- `--seed`: Random seed for reproducibility

### Dataset Structure

The dataset should be organized in one of two ways:

1. Pre-split structure:
```
dataset/
├── train/
│   ├── class1/
│   └── class2/
├── validation/  (optional)
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

2. Single directory structure (will be automatically split):
```
dataset/
├── class1/
└── class2/
```

### Batch Processing

The project includes SLURM script templates for batch processing on HPC clusters. See `inceptionv3.sh` for an example.

## Evaluation

The system supports two evaluation modes:

1. **Closed-set**: All classes in the test set are present in the training set
2. **Open-set**: Test set may contain previously unseen classes

Evaluation metrics include:
- Mean Average Precision (mAP@1, mAP@5)
- Training and evaluation time
- Validation loss

## Output

The experiment results are saved in a ZIP file containing:
- Model summary
- Training parameters
- Results in JSON and CSV formats
- Training logs
- Dataset split information
- Trained model (if --save_model is True)

## GPU Support

The system automatically detects and uses available GPU resources. For optimal performance, ensure CUDA toolkit and appropriate drivers are installed.
