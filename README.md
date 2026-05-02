# MNIST Image Recognition Experiments

A simple PyTorch project comparing three neural network architectures (MLP, CNN, and a Mini Vision Transformer) on the MNIST handwritten digit dataset, with and without data augmentation.

## Setup and Installation

You will need Python 3 and the PyTorch libraries installed. It is highly recommended to use a virtual environment.

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the environment (Mac/Linux)
source .venv/bin/activate

# 3. Install required libraries
pip install torch torchvision
```

## Experiments Results

### Running with: Standard Data

```bash
Training MLP...
MLP Results -> Test Loss: 0.0785, Test Accuracy: 97.59% (Time: 14.4s)

Training CNN...
CNN Results -> Test Loss: 0.0315, Test Accuracy: 98.97% (Time: 68.2s)

Training MiniViT...
MiniViT Results -> Test Loss: 0.0801, Test Accuracy: 97.51% (Time: 79.0s)
```

### Running with: Augmented Data

```bash
Training MLP...
MLP Results -> Test Loss: 0.0820, Test Accuracy: 97.47% (Time: 25.5s)

Training CNN...
CNN Results -> Test Loss: 0.0318, Test Accuracy: 98.85% (Time: 77.3s)

Training MiniViT...
MiniViT Results -> Test Loss: 0.0834, Test Accuracy: 97.41% (Time: 87.1s)
```
