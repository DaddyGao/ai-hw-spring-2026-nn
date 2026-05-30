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

## Analysis

### The Contenders
* **MLP (Multi-Layer Perceptron):** The baseline. Flattens the image into a 1D array.
* **CNN (Convolutional Neural Net):** The vision specialist. Uses 2D sliding filters.
* **MiniViT (Vision Transformer):** The modern approach. Treats the image as a sequence of "patches."

**Speaker Notes:**
> "I tested three distinct architectures to see how they handle spatial data. First, a shallow MLP. It’s fast, but it flattens the 28x28 image, meaning it loses track of which pixels are next to each other. Second, a simple CNN. This is the traditional king of image data because its convolutional filters naturally detect 2D shapes and edges. Finally, a miniaturized Vision Transformer. Instead of pixels or filters, it breaks the image into patches and uses an attention mechanism to figure out how the patches relate to one another."

---

### The Results - Standard Data
* **MLP:** 97.59% (14.4s)
* **CNN:** **98.97%** (68.2s)
* **MiniViT:** 97.51% (79.0s)

**Speaker Notes:**
> "Let’s look at the baseline results after just 5 epochs of training. As expected, the CNN won handily, nearly hitting 99% accuracy. Because it has a built-in understanding of 2D space, it learns visual features very efficiently. The MLP and MiniViT tied at around 97.5%. However, look at the time difference! The MLP achieved that accuracy in just 14 seconds, while the Transformer took nearly 80 seconds. This highlights a known trait of Transformers: they are computationally heavy and usually require massive datasets to beat CNNs. On a small dataset like MNIST, the CNN is vastly superior in both speed and accuracy."

---

### Analysis - The Augmentation Paradox
* **MLP:** 97.59% -> 97.47%
* **CNN:** 98.97% -> 98.85%
* **MiniViT:** 97.51% -> 97.41%
* *Highlight Question:* "Why did accuracy drop?"

**Speaker Notes:**
> "Next, I applied data augmentation—randomly rotating and shifting the training images to make the models more robust. But the results were surprising. Across the board, accuracy actually went *down* slightly, and training time went up significantly by about 10 to 15 seconds per model. 
>
> Why did this happen? It comes down to training duration and dataset distribution. First, augmentation makes the training task much harder. Because I only trained for 5 epochs, the models didn't have enough time to fully converge on this harder task. If we ran this for 50 epochs, the augmented models would likely pull ahead. Second, the standard MNIST training set is identical to the test set—perfectly centered digits. By augmenting the training data, we actually pushed the training distribution slightly away from the test distribution in the short term."

---

### Conclusion & Takeaways
1. CNNs remain the most efficient architecture for small/medium image datasets.
2. Transformers require more data and compute to show their true power.
3. Data Augmentation requires longer training schedules to yield benefits.

**Speaker Notes:**
> "To wrap up, this experiment yielded three great insights. First, CNNs are still the absolute best tool for small-scale image recognition. They give the best accuracy and are highly efficient. Second, while Vision Transformers are the cutting edge of AI right now, they are overkill for simple tasks and require much more compute to reach the same baseline. Finally, data augmentation is not a magic bullet. If you make the training data more chaotic, you must give the neural network more time to make sense of that chaos. Thank you, I'd be happy to take any questions."
