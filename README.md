# Deep Learning Training Design Space — Colab Portfolio

> **Course Assignment** | Executed Jupyter Notebooks with Video Walkthroughs  
> All notebooks are fully executed with outputs. Each has a companion YouTube video walkthrough.

---

##  Table of Contents

| # | Notebook | Topics | Video |
|---|----------|--------|-------|
| 1 | [Activation Functions](#1-activation-functions-for-deep-learning) | Sigmoid, ReLU, GELU, Swish, Mish | [▶ Watch](https://drive.google.com/file/d/12-PMbb0uV2npDB2n37ZYQcO889uRjEDB/view?usp=sharing) |
| 2 | [CNN Fundamentals](#2-convolutional-neural-networks-fundamentals) | Conv layers, Pooling, Architectures, Visualization | [▶ Watch](https://drive.google.com/file/d/1N00MOt7_qwtHzT5e-AuDb2Y5-LLy9Q1N/view?usp=sharing) |
| 3 | [Hyperparameter Tuning](#3-hyperparameter-tuning-for-deep-learning) | Grid Search, Random Search, Bayesian Opt, Optuna | [▶ Watch](https://drive.google.com/file/d/1iurxrc5TD97q0sByQbRJ8CqRatRTzw6R/view?usp=sharing) |
| 4 | [Classification Metrics](#4-classification-metrics--fundamentals) | Confusion Matrix, Precision, Recall, F1, ROC-AUC | [▶ Watch](https://drive.google.com/file/d/1yAQue_yeKUg2jXwNARcNlgvu3iZN07tB/view?usp=sharing) |
| 5 | [Modern CNN Architectures](#5-modern-cnn-architectures--transfer-learning) | ResNet, EfficientNet, Transfer Learning | [▶ Watch](https://drive.google.com/file/d/1mfLQPjrAfx-TBeztV37NBnq9wWA5mz5_/view?usp=sharing) |
| 6 | [Optimizers for Deep Learning](#6-optimizers-for-deep-learning) | SGD, Momentum, Adam, AdamW, LR Schedules | [▶ Watch](https://drive.google.com/file/d/1kwR48gxykPCd7CIgYVCT80bDY1i00fdc/view?usp=sharing) |

---

##  Repository Structure

```
deep-learning-design-space/
│
├── README.md                                         ← You are here
│
├── Colabs/
│   │   └── final_activation_functions_tutorial.ipynb
│   │   └── final_cnn_fundamentals_tutorial.ipynb
│   │   └── final_hyperparameter_tuning_tutorial.ipynb
│   │   └── final_important_classification_metrics_tutorial.ipynb 
│   │   └── final_modern_cnn_architectures_tutorial.ipynb
│       └── final_optimizers_deep_learning_tutorial.ipynb
```

---

## 1. Activation Functions for Deep Learning


### What This Notebook Covers

A comprehensive guide to understanding, visualizing, and choosing activation functions for neural networks — from first principles to modern state-of-the-art.

| Part | Topics |
|------|--------|
| **Part I** | Why Activation Functions Matter — The Problem with Linear Networks |
| **Part II** | Classic Activations — Sigmoid and Tanh |
| **Part III** | The ReLU Revolution — ReLU and the Dying ReLU Problem |
| **Part IV** | ReLU Variants — Leaky ReLU, PReLU, ELU, SELU |
| **Part V** | Modern Activations — GELU (used in GPT), Swish/SiLU, Mish |
| **Part VI** | Output Layer Activations — Sigmoid, Softmax, Linear |
| **Part VII** | Practical Guide: How to Choose the Right Activation |
| **Part VIII** | Hands-On Comparison Across All Activations |

### Key Concepts Demonstrated
- Mathematical intuition behind each activation function
- Visualization of gradients and the vanishing gradient problem
- Benchmark comparison of training speed and accuracy
- Decision framework: hidden layers vs output layers

### Libraries Used
`numpy` · `matplotlib` · `torch` · `scikit-learn`

---

## 2. Convolutional Neural Networks Fundamentals



### What This Notebook Covers

A visual, from-first-principles journey through CNNs — understanding why they work, how they work, and how to build them in PyTorch.

| Chapter | Topics |
|---------|--------|
| **Chapter 1** | The Convolution Operation — why not fully connected layers for images |
| **Chapter 2** | Building CNN Layers in PyTorch — `nn.Conv2d`, output size formula, padding |
| **Chapter 3** | Pooling — Max pooling, Average pooling, spatial downsampling |
| **Chapter 4** | Complete CNN Architectures — feature pyramids, the classic pattern |
| **Chapter 5** | Training CNNs — data augmentation strategies |
| **Chapter 6** | Visualizing What CNNs Learn — feature maps, filters |
| **Chapter 7** | Production-Ready Image Classifier on CIFAR-10 |

### Key Concepts Demonstrated
- Parameter efficiency of convolutions vs fully connected layers
- How filters detect edges, textures, and shapes
- Feature map visualization to understand "what the network sees"
- Data augmentation to prevent overfitting
- End-to-end model training and evaluation

### Libraries Used
`torch` · `torchvision` · `matplotlib` · `numpy`

---

## 3. Hyperparameter Tuning for Deep Learning


### What This Notebook Covers

A complete guide from manual tuning all the way to automated Bayesian optimization using Optuna.

| Part | Topics |
|------|--------|
| **Part I** | Introduction — Parameters vs Hyperparameters |
| **Part II** | Key Hyperparameters — learning rate, batch size, layers, dropout |
| **Part III** | Manual Tuning — Learning Rate Finder technique |
| **Part IV** | Grid Search — exhaustive search, pros and cons |
| **Part V** | Random Search — why it beats grid search (Bergstra & Bengio, 2012) |
| **Part VI** | Bayesian Optimization — surrogate models and acquisition functions |
| **Part VII** | Optuna Tutorial — modern automated hyperparameter optimization |
| **Part VIII** | Best Practices — efficient strategy and common mistakes |

### Key Concepts Demonstrated
- The difference between model parameters and hyperparameters
- Learning rate finder: automated LR range test
- Why random search is more efficient than grid search
- Bayesian optimization: smarter than exhaustive search
- Optuna integration with PyTorch — define-by-run API

### Libraries Used
`torch` · `numpy` · `matplotlib` · `scikit-learn` · `optuna`

---

## 4. Classification Metrics — Fundamentals



### What This Notebook Covers

A complete, progressive guide to classification evaluation metrics — from confusion matrices to ROC-AUC, with real-world intuition for each metric.

| Task | Topics |
|------|--------|
| **Task 1** | Setup & The Classification Problem — synthetic dataset, train/test split |
| **Task 2** | The Confusion Matrix — TP, TN, FP, FN, Type I & II errors |
| **Task 3** | Accuracy & Its Limitations — the accuracy paradox, balanced accuracy |
| **Task 4** | Precision & Recall — trade-off, when to prioritize each |
| **Task 5** | F1 Score & F-beta — harmonic mean, weighted variants |
| **Task 6** | ROC Curve & AUC — threshold analysis, multi-class ROC |
| **Task 7** | Precision-Recall Curve — when AUC is misleading (imbalanced data) |
| **Task 8** | Multi-class Metrics — macro, micro, weighted averaging |
| **Task 9** | MLOps-Ready Evaluation — full model report card |

### Key Concepts Demonstrated
- When accuracy is misleading (imbalanced datasets — fraud, cancer detection)
- The precision-recall tradeoff and how to tune the threshold
- ROC-AUC vs PR-AUC: which to use for imbalanced classes
- Macro vs micro vs weighted averaging for multi-class problems
- Building a complete evaluation dashboard

### Libraries Used
`scikit-learn` · `matplotlib` · `seaborn` · `numpy` · `pandas`

---

## 5. Modern CNN Architectures & Transfer Learning


>  **Note:** Per assignment instructions, this notebook is covered **at a high level only**.

### What This Notebook Covers

High-level overview of the evolution of CNN architectures from ResNet to EfficientNet, with hands-on transfer learning.

| Chapter | Topics |
|---------|--------|
| **Chapter 1** | The Vanishing Gradient Problem — why deeper ≠ better pre-2015 |
| **Chapter 2** | ResNet — residual connections, skip connections, identity shortcut |
| **Chapter 3** | Advanced Building Blocks — depthwise separable convolutions, squeeze-and-excitation |
| **Chapter 4** | EfficientNet — compound scaling of depth, width, and resolution |
| **Chapter 5** | Transfer Learning — feature extraction vs fine-tuning |
| **Chapter 6** | Fine-Tuning Strategies — progressive unfreezing, discriminative LRs |

### Key Concepts Demonstrated (High-Level)
- **ResNet insight:** Instead of learning H(x), learn the residual F(x) = H(x) - x
- **EfficientNet insight:** Scale depth, width, and resolution together systematically
- **Transfer Learning:** Pretrained ImageNet weights → adapt to your domain in minutes
- Feature extraction (freeze backbone) vs fine-tuning (unfreeze layers gradually)

### Libraries Used
`torch` · `torchvision` · `matplotlib` · `numpy`

---

## 6. Optimizers for Deep Learning


### What This Notebook Covers

A deep dive into the full family of deep learning optimizers — from vanilla gradient descent all the way to modern methods like AdamW and SAM.

| Part | Topics |
|------|--------|
| **Part I** | Introduction — Why Optimizers Matter, Loss Landscapes, Contour Plots |
| **Part II** | Gradient Descent Fundamentals — gradients, the update rule |
| **Part III** | Problems with Vanilla GD — ravines, saddle points |
| **Part IV** | Stochastic Gradient Descent — mini-batch GD, batch size effects |
| **Part V** | Momentum Methods — SGD with momentum, Nesterov Accelerated Gradient |
| **Part VI** | Adaptive Methods — AdaGrad, RMSprop |
| **Part VII** | Adam Family — Adam, AdamW (weight decay done right) |
| **Part VIII** | Learning Rate Schedules — step decay, cosine annealing, warmup |
| **Part IX** | Practical PyTorch Comparison — training real networks with each optimizer |
| **Part X** | Modern Optimizers — SAM, Lion, LARS/LAMB |

### Key Concepts Demonstrated
- 3D loss landscape and contour plot visualizations
- Why momentum helps escape ravines and saddle points
- The per-parameter adaptive learning rate insight of Adam
- Why AdamW fixes Adam's weight decay bug
- LR warm-up and cosine annealing schedules
- Optimizer evolution timeline from 1951 to present

### Libraries Used
`torch` · `torchvision` · `numpy` · `matplotlib` · `scikit-learn`

---

##  Learning Summary

By completing all six notebooks, the following core concepts of the deep learning training design space are covered:

```
Deep Learning Training Design Space
│
├── Architecture Choices
│   ├── Activation Functions     → How neurons fire
│   ├── CNN Fundamentals         → Spatial feature extraction
│   └── Modern CNN Architectures → ResNet, EfficientNet, Transfer Learning
│
├── Optimization
│   ├── Optimizers               → How the model learns (SGD → Adam)
│   └── Hyperparameter Tuning    → How to find the best settings
│
└── Evaluation
    └── Classification Metrics   → How to measure success honestly
```

---



*Last updated: April 2026*
