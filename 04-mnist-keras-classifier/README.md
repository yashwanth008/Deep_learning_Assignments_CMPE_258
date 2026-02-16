# MNIST Neural Network Classifier

**Video Walkthrough** [https://drive.google.com/file/d/1k63xUwwYJO2xY0zwSGgGEYSWk4pQ-Gs4/view?usp=sharing]
##  Overview
A comprehensive implementation of a Convolutional Neural Network (CNN) for MNIST digit classification using Keras/TensorFlow. Features complete metrics, visualizations, and professional-grade evaluation.

##  Features

### Model Architecture
- **Deep CNN** with 3 convolutional blocks
- **Batch Normalization** for stable training
- **Dropout Regularization** to prevent overfitting
- **Dense Layers** for final classification
- **10-class Softmax** output layer

### Training Features
- **Data Augmentation Ready** - Extensible design
-  **Validation Split** - 10% of training data
-  **Early Stopping** - Prevents overfitting
-  **Learning Rate Scheduling** - Adaptive optimization
-  **Model Checkpointing** - Saves best model
-  **Reproducible Results** - Fixed random seeds

### Comprehensive Metrics
- **Accuracy & Loss Curves** - Training visualization
- **Confusion Matrix** - Both raw and normalized
- **Classification Report** - Precision, recall, F1-score per class
- **ROC Curves** - Multi-class ROC analysis
- **Per-Class Metrics** - Detailed performance breakdown
- **Misclassification Analysis** - Error visualization
- **Sample Predictions** - Visual validation

### Visualizations Generated
1. Sample MNIST digits
2. Model architecture diagram
3. Training history (accuracy & loss)
4. Confusion matrices
5. Per-class performance charts
6. ROC curves for all 10 digits
7. Sample predictions
8. Misclassified examples
9. Summary report (text file)

##  Technologies Used

### Core Frameworks
- **TensorFlow 2.13+** - Deep learning framework
- **Keras** - High-level neural networks API
- **Python 3.8+** - Programming language

### Data & Visualization
- **NumPy** - Numerical computing
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualization
- **Pandas** - Data manipulation
- **Scikit-learn** - Metrics and evaluation

### AI Tools Used
- **Google Antigravity** - AI-assisted development
- **Claude Code** - Code optimization
- **wshobson Skills** - Best practices

## 

**YouTube Link:** [Add your video link here]

### Video Contents:
1. **Introduction** (0:00 - 2:00)
   - Project overview
   - MNIST dataset explanation
   - Expected outcomes

2. **Code Walkthrough** (2:00 - 10:00)
   - Data loading and preprocessing
   - Model architecture explanation
   - Training configuration
   - Metrics implementation

3. **Execution Demo** (10:00 - 15:00)
   - Running the classifier
   - Live training process
   - Real-time metrics display

4. **Results Analysis** (15:00 - 20:00)
   - Reviewing all visualizations
   - Confusion matrix interpretation
   - ROC curve analysis
   - Performance evaluation

##  Installation & Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# pip package manager
pip --version
```

### Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/[your-username]/ai-coding-assignment-2026.git
   cd ai-coding-assignment-2026/04-mnist-keras-classifier
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (Mac/Linux)
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Classifier**
   ```bash
   python mnist_classifier.py
   ```

5. **View Results**
   ```bash
   # All outputs are saved in the 'outputs/' directory
   ls outputs/
   ```

##  Project Structure

```
04-mnist-keras-classifier/
├── mnist_classifier.py       # Main classifier script
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── outputs/                 # Generated outputs (created on run)
    ├── 01_sample_images.png
    ├── 02_model_architecture.png
    ├── 03_training_history.png
    ├── 04_classification_report.csv
    ├── 05_confusion_matrix.png
    ├── 06_per_class_metrics.png
    ├── 07_roc_curves.png
    ├── 08_sample_predictions.png
    ├── 09_misclassified.png
    ├── best_model.h5
    ├── mnist_model_final.h5
    └── summary_report.txt
```

##  Model Architecture

### Network Design

```
Input: (28, 28, 1) - Grayscale images

Conv Block 1:
├── Conv2D(32, 3x3) + ReLU + BatchNorm
├── Conv2D(32, 3x3) + ReLU + BatchNorm
├── MaxPooling(2x2)
└── Dropout(0.25)

Conv Block 2:
├── Conv2D(64, 3x3) + ReLU + BatchNorm
├── Conv2D(64, 3x3) + ReLU + BatchNorm
├── MaxPooling(2x2)
└── Dropout(0.25)

Conv Block 3:
├── Conv2D(128, 3x3) + ReLU + BatchNorm
├── MaxPooling(2x2)
└── Dropout(0.4)

Dense Layers:
├── Flatten()
├── Dense(256) + ReLU + BatchNorm + Dropout(0.5)
├── Dense(128) + ReLU + BatchNorm + Dropout(0.5)
└── Dense(10, Softmax)

Output: 10 classes (digits 0-9)
```

### Key Components

**Convolutional Layers:**
- Extract spatial features from images
- Progressive feature map increase (32 → 64 → 128)
- 3x3 kernels for local pattern recognition

**Batch Normalization:**
- Stabilizes training
- Allows higher learning rates
- Reduces internal covariate shift

**Dropout:**
- Prevents overfitting
- Increases 0.25 → 0.4 → 0.5 for dense layers
- Forces network to learn robust features

**Pooling:**
- Reduces spatial dimensions
- Provides translation invariance
- Decreases computational cost

##  Training Configuration

### Hyperparameters

```python
Optimizer: Adam
Learning Rate: 0.001 (adaptive with ReduceLROnPlateau)
Loss Function: Sparse Categorical Crossentropy
Batch Size: 128
Epochs: 30 (with early stopping)
Validation Split: 10%
```

### Callbacks

1. **EarlyStopping**
   - Monitor: val_loss
   - Patience: 5 epochs
   - Restores best weights

2. **ReduceLROnPlateau**
   - Reduces LR by factor 0.5
   - Patience: 3 epochs
   - Minimum LR: 1e-7

3. **ModelCheckpoint**
   - Saves best model based on val_accuracy
   - Automatic overwrites

##  Expected Results

### Performance Metrics

Typical results after training:

```
Test Accuracy: ~99.3% - 99.6%
Test Loss: ~0.02 - 0.03

Per-Class Metrics:
- Precision: 0.99+
- Recall: 0.99+
- F1-Score: 0.99+
```

### Training Time

- **CPU:** ~30-45 minutes (30 epochs)
- **GPU:** ~5-10 minutes (30 epochs)
- **Memory:** ~2-3 GB RAM

##  Understanding the Outputs

### 1. Sample Images (01_sample_images.png)
Shows 10 random training examples with their labels

### 2. Model Architecture (02_model_architecture.png)
Visual diagram of the entire network structure

### 3. Training History (03_training_history.png)
Plots showing:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Helps identify overfitting/underfitting

### 4. Classification Report (04_classification_report.csv)
Detailed metrics for each digit class:
- Precision: Correctness of positive predictions
- Recall: Coverage of actual positives
- F1-Score: Harmonic mean of precision and recall
- Support: Number of true instances

### 5. Confusion Matrix (05_confusion_matrix.png)
Two matrices:
- **Left (Counts):** Raw prediction counts
- **Right (Normalized):** Percentages per class
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

### 6. Per-Class Metrics (06_per_class_metrics.png)
Bar chart comparing precision, recall, and F1-score across all digits

### 7. ROC Curves (07_roc_curves.png)
Receiver Operating Characteristic curves for each digit
- Shows true positive vs false positive rates
- Area Under Curve (AUC) near 1.0 indicates excellent performance

### 8. Sample Predictions (08_sample_predictions.png)
15 test images with true and predicted labels
- Green titles: Correct predictions
- Red titles: Incorrect predictions

### 9. Misclassified Examples (09_misclassified.png)
Shows examples where the model failed
- Helps identify difficult cases
- Useful for model improvement

##  Code Structure

### Main Components

```python
class MNISTClassifier:
    """Complete MNIST Classifier"""
    
    __init__()                      # Initialize classifier
    load_and_preprocess_data()      # Load MNIST dataset
    build_model()                   # Create CNN architecture
    train_model()                   # Train with callbacks
    evaluate_model()                # Comprehensive evaluation
    save_model()                    # Save trained model
    generate_summary_report()       # Create text report
    
    # Visualization methods
    _visualize_samples()
    _plot_training_history()
    _plot_confusion_matrix()
    _plot_per_class_metrics()
    _plot_roc_curves()
    _visualize_predictions()
    _visualize_misclassified()
```

### Key Functions

**Data Processing:**
- Normalization: Scales pixels to [0, 1]
- Reshaping: Adds channel dimension
- Train/Val split: 90/10 split

**Model Building:**
- Sequential API for linear stack
- Functional API compatible design

**Training:**
- Batch processing for memory efficiency
- Callback integration for smart training

**Evaluation:**
- scikit-learn metrics integration
- Custom visualization functions

##  Customization Options

### Change Model Architecture

```python
# In build_model() method
# Modify layers, filters, or depth

# Example: Add more convolutional layers
layers.Conv2D(256, (3, 3), activation='relu', padding='same')
```

### Adjust Training Parameters

```python
# In train_model() method
classifier.train_model(
    epochs=50,           # More epochs
    batch_size=64        # Smaller batches
)
```

### Modify Callbacks

```python
# Adjust early stopping patience
EarlyStopping(patience=10)  # More patient

# Change learning rate schedule
ReduceLROnPlateau(factor=0.2, patience=5)
```

