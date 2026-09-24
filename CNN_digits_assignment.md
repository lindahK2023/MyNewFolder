# CNN exercise: predict handwritten digits

**Response variable:** digit label (0–9). **Covariates:** the 64 pixel intensities in an 8×8 image.

This uses the built-in handwritten digits dataset from scikit-learn, so no class data directory or external download is needed. The convolution, pooling, flattening and dense layer match slide 31 of *C04_DL_CNN*. Train and evaluate on your laptop's CPU.

**Before running:** In PyCharm's Terminal at the `(.venv) PS ...>` prompt, run `python -m pip install scikit-learn matplotlib ipykernel`. PyTorch is already installed in this project. Open this notebook in PyCharm, select the project's `.venv` as its kernel, and run the cells from top to bottom.

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
print("PyTorch:", torch.__version__)
print("Compute device: CPU")
```

## 1. Inspect the data

Each example is a grayscale 8×8 image. Pixel brightness is between 0 and 16. The label is the digit shown in the image. We divide brightness by 16 to put inputs between 0 and 1.

```python
digits = load_digits()
X = (digits.images / 16.0).astype("float32")
y = digits.target.astype("int64")
print("Images:", X.shape, "Labels:", y.shape)
print("Example label:", y[0])
plt.imshow(X[0], cmap="gray_r", vmin=0, vmax=1)
plt.title(f"Example digit: {y[0]}")
plt.axis("off")
plt.show()
```

## 2. Hold out a test set

The model learns from 80% of the images. We keep 20% unseen until evaluation. `stratify=y` keeps the digit classes balanced in both sets. The extra channel dimension gives PyTorch the shape `[number of images, 1, 8, 8]`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_images = torch.from_numpy(X_train).unsqueeze(1)
test_images = torch.from_numpy(X_test).unsqueeze(1)
train_labels = torch.from_numpy(y_train)
test_labels = torch.from_numpy(y_test)
train_loader = DataLoader(
    TensorDataset(train_images, train_labels), batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(test_images, test_labels), batch_size=128, shuffle=False
)
print("Training:", train_images.shape, "Test:", test_images.shape)
```

## 3. Build the CNN

`Conv2d(1, 8, 3)`: eight 3×3 filters turn 1×8×8 into 8×6×6. ReLU keeps positive activations. `MaxPool2d(2)` turns 8×6×6 into 8×3×3. Flatten gives 72 features, and the dense layer returns ten class scores. These scores are called logits; `CrossEntropyLoss` handles the conversion internally.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 3 * 3, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc1(x)

model = SimpleCNN()
print(model)
print("Output shape for 4 images:", model(train_images[:4]).shape)
```

## 4. Train

An epoch is one pass through the training images. For each batch: predict, calculate cross entropy loss, clear old gradients, calculate new gradients with `backward()`, and update weights with Adam.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
history = []

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
    mean_loss = running_loss / len(train_images)
    history.append(mean_loss)
    print(f"Epoch {epoch + 1:2d}: training loss = {mean_loss:.4f}")

plt.plot(range(1, len(history) + 1), history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training progress")
plt.show()
```

## 5. Evaluate on images the model has never seen

Accuracy is the fraction of held-out images classified correctly. The confusion matrix shows which digit pairs get mixed up. `model.eval()` puts the network in prediction mode and `torch.no_grad()` stops gradient calculations.

```python
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        predictions.append(model(images).argmax(dim=1))
predictions = torch.cat(predictions).numpy()
accuracy = (predictions == y_test).mean()
print(f"Test accuracy: {accuracy:.1%} ({(predictions == y_test).sum()}/{len(y_test)})")

cm = confusion_matrix(y_test, predictions, labels=range(10))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues", values_format="d")
plt.title("Held-out test images")
plt.show()
```

```python
fig, axes = plt.subplots(1, 5, figsize=(11, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i], cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"True: {y_test[i]}\nPred: {predictions[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Six points for a five-minute presentation

1. **Question:** Can an 8×8 image's pixels predict which digit (0–9) it shows?
2. **Data:** The scikit-learn digits dataset has 1,797 grayscale images; 80% train and 20% test.
3. **Model:** 3×3 convolution with eight filters, ReLU, max pooling, flatten, ten output scores, matching slide 31.
4. **Training:** Cross entropy loss, Adam optimiser, ten epochs. Show the loss graph.
5. **Result:** Read the **actual test accuracy from your output**, and show the confusion matrix and example images. Do not invent an accuracy before running.
6. **Limit:** These tiny handwriting images are a practice dataset. The satellite task needs spatial images, dates, consistent preprocessing, and a clearly defined target.

**When the class data arrives:** identify whether it is 1D sequences or 2D images, choose a response variable, and replace the data-loading and input-shape cells. Keep the train/test split and evaluation logic; adapt the convolution to the data shape. For time series, split by time where appropriate to avoid leakage.

## Results from the run in PyCharm

The uploaded notebook copy contains no saved outputs. These results were recorded from the screenshots of the run:

- Dataset: 1,797 handwritten digit images, each 8 × 8 pixels, with one correct digit label per image.
- Split: 1,437 training images and 360 test images.
- Training loss: 1.9520 at epoch 1 and 0.1011 at epoch 10.
- Test accuracy: **96.7% (348 correct out of 360)**.
- The five displayed examples were correctly predicted as 5, 2, 8, 1, and 7.

The training graph, confusion matrix, and example image plot are generated by the code blocks when you run the notebook; they are not embedded in this Markdown file.
##Notes: testing branches 
