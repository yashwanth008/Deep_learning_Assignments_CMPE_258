# Deep Learning Fundamentals

A complete hands-on series covering the mathematics, implementation, and frameworks behind modern deep learning — from visual intuition through production-ready code.

Each notebook is fully executed with outputs and paired with a video explanation. Built as a course project demonstrating both conceptual understanding and practical skill.

---

## Interactive Demo

### Neural Network Playground
**[Open Playground →](https://dlmastery.github.io/neural-network-playground/)**

An interactive browser-based visualization where you can watch a neural network learn in real time. No code required — just drag sliders and observe.

**What it shows:**
- How decision boundaries form and evolve during training
- Why depth matters: XOR is unsolvable by a single layer but trivial with two
- The effect of different activation functions (ReLU, Tanh, Sigmoid)
- How learning rate controls convergence — too small is slow, too large diverges
- Why regularization smooths boundaries and prevents overfitting

**Recommended experiments:**
1. Load the XOR dataset with 1 hidden layer → watch it fail → add a second layer → watch it succeed
2. Load the Spiral dataset with no regularization → observe overfitting → add L2 = 0.003
3. Try learning rates 0.001, 0.03, and 1.0 on the Circle dataset and compare convergence

> **Video explanation:** [ Watch Video 0 →](https://drive.google.com/file/d/1wGTrxXpM7hpUEaTlQ6LiZW5MYYAhdF-k/view?usp=sharing)

---

## Notebook Series

All notebooks are in the directory, organized by topic. Each links to a video walkthrough.

---

###  Part 1: Mathematical Foundations

These four notebooks build the math that underlies every deep learning framework. Understanding them means you can debug issues that framework users can't, and implement ideas that don't exist yet.

---

#### 1. NumPy Foundations for Deep Learning
**Notebook:** numpy_foundations_for_deep_learning.

**Video:** [Watch →](https://drive.google.com/file/d/1KILzV1H_vTy4gXuk3VkLk6oAxsvyNk0g/view?usp=sharing)

The starting point. NumPy's array model, broadcasting rules, and vectorized operations are the building blocks every framework is built on.

| Section | What you learn |
|---------|---------------|
| Array creation & shapes | How data is represented: scalars, vectors, matrices, N-D tensors |
| Broadcasting | Why `W @ X + b` works without loops — the most important NumPy concept |
| Matrix multiplication | The core operation of every neural network layer |
| Vectorization | Why NumPy is 80×+ faster than Python loops |
| NN layer simulation | `z = X @ W + b` is a complete neural network layer |

**Key insight:** When you call `output = W @ X + bias`, the `+ bias` is broadcasting — the same mechanism NumPy uses to add a vector to a matrix without copying.

---

#### 2. Linear Algebra for Deep Learning
**Notebook:** linear_algebra_for_deep_learning.ipynb

**Video:** [Watch →](https://drive.google.com/file/d/1-ZtsZQ3OinTV_XyS_rRDqN7bThtOj_4D/view?usp=sharing)

Neural network layers are geometric transformations. This notebook builds the geometric intuition for what matrix multiplication is actually doing to your data.

| Section | What you learn |
|---------|---------------|
| Dot products | How neurons measure input alignment with their weights |
| Matrices as transformations | Each layer rotates/stretches the input space |
| Matrix multiplication | Composing transformations — why depth works |
| Eigenvalues | Why they explain gradient flow and training stability |
| SVD | Matrix decomposition used in dimensionality reduction and attention |

**Key insight:** A neural network doesn't "compute" an output — it transforms the input space, layer by layer, until classes become linearly separable.

---

#### 3. Calculus for Deep Learning
**Notebook:** calculus_for_deep_learning.ipynb

**Video:** [Watch →](https://drive.google.com/file/d/1WaAVlMLeZMfF5cHFwynuLayhLsXrMkUv/view?usp=sharing)

The chain rule is backpropagation. Not metaphorically — the algorithm PyTorch runs when you call `.backward()` is exactly the chain rule applied to a computation graph. This notebook derives it from scratch.

| Section | What you learn |
|---------|---------------|
| Derivatives | The rate of change — what a gradient is |
| Numerical vs analytical | Two ways to compute derivatives and why they match |
| Activation derivatives | Why sigmoid causes vanishing gradients, why ReLU doesn't |
| The chain rule | Propagating gradients through composed functions |
| Gradient descent | Stepping opposite the gradient to minimize loss |
| Full XOR backprop | A complete 2-layer network trained from scratch with pure math |

**Key insight:** `dz2 = a2 - y` — the gradient of cross-entropy + sigmoid is just prediction minus target. Every other line in backpropagation follows from the chain rule.

---

#### 4. Probability for Deep Learning
**Notebook:** probability_for_deep_learning.ipynb

**Video:** [Watch →](https://drive.google.com/file/d/1piwaGDFVb8OrX9_WBiA2tyFFfWPXoVQN/view?usp=sharing)

Neural networks output probability distributions. Understanding the math behind softmax and cross-entropy explains why these functions were chosen and what they're computing.

| Section | What you learn |
|---------|---------------|
| Distributions | Normal distribution for weight initialization |
| Softmax | Converting raw logits to probability distributions |
| Cross-entropy loss | Why `-log(p_correct)` measures prediction quality |
| MLE perspective | Training as maximum likelihood estimation |
| KL divergence | The information-theoretic view of cross-entropy |

**Key insight:** The gradient of cross-entropy loss through softmax simplifies to `predicted - actual`. This clean result is why the softmax + cross-entropy combination is used everywhere.

---

### Part 2: Tensor Libraries

Three notebooks covering the tensor primitives of the three major frameworks. Same concepts, three different APIs.

---

#### 5. PyTorch Tensors: From Zero to Hero
**Notebook:** pytorch_tensors_from_zero_to_hero.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1LAo5CZk5aD6MRcX0ODV1YGS6KOOgf5Mw/view?usp=sharing)

The most comprehensive single-notebook PyTorch reference in this series. 10 parts covering every tensor operation you'll need, culminating in autograd and real training patterns.

| Part | Topics |
|------|--------|
| 1. What is a tensor | 0D–4D tensors, shapes for different data types |
| 2. Creating tensors | torch.tensor, randn, zeros, ones, like variants |
| 3. Shape, dtype, device | The three fundamental attributes of every tensor |
| 4. Indexing & slicing | Basic, fancy, and boolean indexing |
| 5. Basic operations | Arithmetic, math functions, aggregation, broadcasting |
| 6. Reshaping | view, flatten, squeeze, unsqueeze, permute, cat, stack |
| 7. Linear algebra | Matrix multiplication, dot products, norms, decompositions |
| 8. Einstein summation | einsum for attention, batch ops, arbitrary contractions |
| 9. Deep learning ops | Activation functions, softmax, loss functions, normalization |
| 10. Practical patterns | Weight init, autograd, no_grad, one-hot, masking, padding |

**Key insight:** `requires_grad=True` + `.backward()` runs the chain rule automatically through every operation in your forward pass. No manual gradient computation needed.

---

#### 6. TensorFlow Tensor Operations
**Notebook:** tensorflow_tensor_operations_tutorial.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1xhjPeqZfJhovn4gFRTPx2wtpuLeCbfOH/view?usp=sharing)

TensorFlow's tensor API with focus on what's different from PyTorch: GradientTape, tf.Variable, and @tf.function for production deployment.

| Topic | TF API | PyTorch equivalent |
|-------|--------|-------------------|
| Automatic differentiation | `with tf.GradientTape()` | `requires_grad=True` + `.backward()` |
| Trainable parameters | `tf.Variable` | `tensor(requires_grad=True)` |
| JIT compilation | `@tf.function` | `torch.jit.script` |
| GPU placement | Automatic via XLA | `.to('cuda')` |

**Key insight:** GradientTape records operations like a tape recorder. `tape.gradient(loss, variables)` plays it back to compute derivatives — same chain rule, different metaphor.

---

### 🟢 Part 3: Neural Networks from Scratch

Two notebooks that bridge mathematical foundations and framework usage.

---

#### 7. Neural Networks from Scratch
**Notebook:** neural_networks_from_scratch.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1Yu_QkMIu0hSKotX5xgLdu_y4XoZjHgDf/view?usp=sharing)

A complete 2-layer neural network implemented in pure NumPy: forward pass, backpropagation, gradient descent. No frameworks.

```python
# The complete forward pass
z1 = X @ W1 + b1           # Linear transformation
a1 = np.maximum(0, z1)     # ReLU activation
z2 = a1 @ W2 + b2          # Second linear layer
a2 = sigmoid(z2)            # Sigmoid output
```

```python
# The complete backward pass — pure chain rule
dz2 = a2 - y               # Gradient at output
dW2 = (a1.T @ dz2) / m     # Weight gradient (layer 2)
da1 = dz2 @ W2.T           # Propagate error back
dz1 = da1 * (z1 > 0)       # ReLU derivative
dW1 = (X.T @ dz1) / m      # Weight gradient (layer 1)
```

**Result:** Solves XOR from random initialization. Converges to correct predictions [0, 1, 1, 0].

**Key insight:** Every line of backpropagation is one application of the chain rule. PyTorch's `.backward()` runs this same computation automatically for any architecture.

---

#### 8. Why Deep Learning Works: Geometric Intuition
**Notebook:** geometric_intuition.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1q_DXXp61unaGODHCqg2v0_HnEf8eFjM0/view?usp=sharing)

The conceptual bridge: why does any of this work? Visualizations of feature space transformations, decision boundary evolution, and the loss landscape.

| Concept | What the visualizations show |
|---------|------------------------------|
| Feature space warping | Each layer applies a learnable transformation to input space |
| Linear separability | Deep layers warp data until a hyperplane can separate classes |
| Decision boundaries | How they evolve from random to meaningful during training |
| Loss landscape | Non-convex in theory, navigable in practice for high-dim problems |
| Depth vs width | Why hierarchical features require layers, not just neurons |

---

### 🔴 Part 4: Framework Deep Dives

Production-quality implementations using the three major frameworks.

---

#### 9. Keras/TensorFlow Neural Networks Tutorial
**Notebook:** keras_tensorflow_tutorial.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1gBrPA33hc3tdP2QFC83lMuBGRTW0iL-K/view?usp=sharing)

Building and training neural networks with Keras's high-level API: Sequential and Functional models, training loop, evaluation, and essential callbacks.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

---

#### 10. Advanced Keras/TensorFlow Tutorial
**Notebook:** [`colabs/frameworks/keras_tensorflow_advanced.ipynb`](./colabs/frameworks/keras_tensorflow_advanced.ipynb)
**Video:** [Watch →](https://drive.google.com/file/d/1yPDsmkbalByIiaKKtZJB-0jSJbqzjgtx/view?usp=sharing)

Custom training loops, custom layers, tf.data pipelines, and model export for production deployment.

| Topic | What you build |
|-------|---------------|
| Custom training loops | Manual GradientTape training for full control |
| Custom layers | Subclass tf.keras.Layer with custom build() and call() |
| tf.data | Efficient data pipelines with prefetch, cache, shuffle |
| Model saving | SavedModel format for production serving |

---

#### 11. JAX Neural Networks Tutorial
**Notebook:** jax_neural_networks.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1RiJaczFkih7H_xDfQdXscyinXMIlg0Ss/view?usp=sharing)

JAX's functional approach to deep learning: pure functions, explicit PRNG keys, and composable transformations (grad, jit, vmap).

| Transformation | What it does |
|----------------|-------------|
| `grad(f)` | Returns a function that computes the gradient of f |
| `jit(f)` | Compiles f to XLA — fast after first call |
| `vmap(f)` | Vectorizes f over a batch dimension automatically |
| `value_and_grad(f)` | Returns both f(x) and ∇f(x) in one pass |

**Key insight:** These transformations compose: `jit(vmap(grad(loss_fn)))` gives JIT-compiled per-example gradients — difficult in PyTorch, trivial in JAX.

---

#### 12. Advanced JAX Deep Learning
**Notebook:** jax_advanced.ipynb
**Video:** [Watch →](https://drive.google.com/file/d/1JOCyQPU4vEh6lXQVcZE6OzX5WDGupxke/view?usp=sharing)

Advanced JAX: Flax for neural network modules, Optax for functional optimizers, and pmap for multi-device training.

---

