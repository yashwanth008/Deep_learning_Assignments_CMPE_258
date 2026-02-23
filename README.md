# Neural Networks From Scratch
### NumPy · PyTorch · PyTorch Lightning · TensorFlow

> **Course Assignment** | Deep Learning | Spring 2025
> 
> A complete implementation of a 3-layer deep neural network for nonlinear regression built **5 different ways** across 7 Colab notebooks — from raw NumPy with manual backpropagation all the way to high-level Keras APIs.

---

##  Video Walkthroughs



| Colab | Title | Video |
|---|---|---|
| A | NumPy From Scratch | [▶ Watch](https://drive.google.com/file/d/1sWc-oBcBkXUtcDW-L-CnRe7q33T2j2ZQ/view?usp=sharing) |
| B | PyTorch From Scratch | [▶ Watch](https://drive.google.com/file/d/1QU9dUvwQLGjiFpVIucc8L0e-Q92PqeBd/view?usp=sharing) |
| C | PyTorch Class-Based | [▶ Watch](https://drive.google.com/file/d/1fpdvAz5JTAKl9nbfItjJwqdpsqLvKLnU/view?usp=sharing) |
| D | PyTorch Lightning | [▶ Watch](https://drive.google.com/file/d/1lHO1Sd9VZSjScZFKuh0w9hP2lwOu6yNA/view?usp=sharing) |
| E-i - iv| TensorFlow overall | [▶ Watch](https://drive.google.com/file/d/1NjyftUr_SC4U57kEh-F5FYO-qcIswRUT/view?usp=sharing) |


---

## Colab Notebooks

| Colab | Description | Open |
|---|---|---|
| A | NumPy only — manual forward pass, manual backprop, chain rule gradients, `tf.einsum` | 
| B | PyTorch from scratch — `nn.Parameter` weights, autograd backprop, no built-in layers |
| C | PyTorch class-based — `nn.Linear`, `nn.Sequential`, Adam optimizer, LR scheduler | 
| D | PyTorch Lightning — `LightningModule`, `Trainer`, callbacks, mini-batch training | 
| E | TensorFlow — all four variants in one notebook | 

---


##  The Problem

All notebooks solve the same **nonlinear regression** problem.

### Equation
```
y = sin(x1) + x2² + x1·x3
```

### Synthetic Dataset
- **1,000 samples** generated from the equation above
- **3 input variables:** x1, x2, x3 sampled uniformly from [-3, 3]
- **Gaussian noise** (σ = 0.1) added to simulate real-world data
- **4D visualization** using PCA to compress inputs to 2D, output as color

### Network Architecture
```
Input Layer  →  3 neurons  (x1, x2, x3)
Hidden Layer 1  →  16 neurons  +  ReLU
Hidden Layer 2  →   8 neurons  +  ReLU
Output Layer    →   1 neuron   (no activation — regression)
```

---

##  Notebook Descriptions

### Colab A — NumPy From Scratch
**Framework:** NumPy only (+ `tf.einsum` for matrix multiply as required)

The most foundational notebook. Every single operation is implemented by hand:
- Weight initialization using **He initialization**
- Forward pass using `tf.einsum` instead of matrix multiply
- **Manual backpropagation** using the chain rule — every gradient computed by hand
- Weight updates written explicitly: `W = W - lr * dW`
- Loss curve, predicted vs actual scatter, and side-by-side 4D PCA plots

Key concepts demonstrated: chain rule, ReLU derivative as a gate, gradient shapes matching weight shapes.

---

### Colab B — PyTorch From Scratch
**Framework:** PyTorch — `nn.Parameter` only, no `nn.Linear`

Rebuilds Colab A in PyTorch without using any built-in layer functionality:
- Weights declared as `nn.Parameter` tensors (registers them for gradient tracking)
- Forward pass written manually using `torch.matmul`
- **PyTorch autograd** handles backpropagation — `loss.backward()` replaces manual chain rule
- SGD optimizer handles weight updates — `optimizer.step()` replaces manual `W -= lr * dW`

Key difference from Colab A: we write the forward pass, PyTorch writes the backward pass.

---

### Colab C — PyTorch Class-Based
**Framework:** PyTorch — full `nn.Module` with built-in layers

Production-style PyTorch using all standard tools:
- `nn.Linear` replaces manual weight declaration
- `nn.Sequential` replaces manual layer chaining
- **Adam optimizer** replaces SGD — adapts learning rate per parameter
- **ReduceLROnPlateau scheduler** halves learning rate when training stalls
- Learning rate curve plotted alongside loss curve

Key difference from Colab B: built-in layers + smarter optimization = cleaner code + better results.

---

### Colab D — PyTorch Lightning
**Framework:** PyTorch Lightning

Same architecture as Colab C wrapped in a `LightningModule`:
- `training_step()` replaces the manual training loop
- `configure_optimizers()` returns optimizer + scheduler
- `Trainer.fit()` replaces the entire `for epoch in range(...)` loop
- **Mini-batch training** via `DataLoader` (batch size 64)
- Lightning handles `zero_grad`, `backward`, `optimizer.step` automatically

Key difference from Colab C: same math, all training boilerplate removed.

---

### Colab E — TensorFlow (All Four Variants)

One notebook containing four progressively higher-level implementations:

**E-i: TF Low Level**
Raw TensorFlow with no Keras. Weights as `tf.Variable`, forward pass with `tf.matmul`,
gradients via `tf.GradientTape`. TF equivalent of Colab B.

**E-ii: Keras Layers + Manual Loop**
Uses `keras.Dense` layers for model definition but still writes the training loop
manually with `GradientTape`. Middle ground between E-i and E-iii.

**E-iii: Keras Functional API**
Builds the model by calling layers as functions on tensors — explicit graph wiring.
Supports skip connections and multi-input architectures unlike Sequential.
Uses `model.compile()` and `model.fit()` — no manual loop.

**E-iv: Keras Sequential + Callbacks**
Simplest possible TensorFlow code. Adds production features:
- **EarlyStopping** — stops when validation loss stalls, restores best weights
- **ReduceLROnPlateau** — halves learning rate on plateau
- **Validation split** — 10% held out to monitor overfitting
- Train vs validation loss plotted together

---

##  Results Summary

| Colab | Framework | Approach | R² Score |
|---|---|---|---|
| A | NumPy | Manual backprop + `tf.einsum` | ~0.92 |
| B | PyTorch | Manual weights + autograd | ~0.92 |
| C | PyTorch | Built-in layers + Adam | ~0.95 |
| D | PyTorch Lightning | LightningModule + Trainer | ~0.95 |
| E-i | TensorFlow | Raw `tf.Variable` + GradientTape | ~0.93 |
| E-ii | TensorFlow | Keras layers + manual loop | ~0.94 |
| E-iii | TensorFlow | Functional API + model.fit | ~0.95 |
| E-iv | TensorFlow | Sequential + EarlyStopping | ~0.95 |

> All results are on the same 1,000-sample synthetic dataset.
> R² measures how much variance the network explains — 1.0 is perfect.

---

