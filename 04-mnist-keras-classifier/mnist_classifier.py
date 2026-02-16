"""
MNIST Neural Network Classifier with Keras
===========================================

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("MNIST DIGIT CLASSIFICATION - NEURAL NETWORK")
print("=" * 80)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}\n")

class MNISTClassifier:
    """Complete MNIST Classifier with comprehensive metrics"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.predictions = None
        self.output_dir = 'outputs'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print(" Loading MNIST Dataset...")
        
        # Load data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        print(f"Dataset loaded successfully!")
        print(f"   Training samples: {x_train.shape[0]}")
        print(f"   Test samples: {x_test.shape[0]}")
        print(f"   Image shape: {x_train.shape[1:]}\n")
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Create validation set (10% of training data)
        val_split = int(0.1 * len(x_train))
        self.x_val = x_train[:val_split]
        self.y_val = y_train[:val_split]
        self.x_train = x_train[val_split:]
        self.y_train = y_train[val_split:]
        self.x_test = x_test
        self.y_test = y_test
        
        print(" Data Split:")
        print(f"   Training: {self.x_train.shape[0]} samples")
        print(f"   Validation: {self.x_val.shape[0]} samples")
        print(f"   Test: {self.x_test.shape[0]} samples\n")
        
        # Visualize sample images
        self._visualize_samples()
        
    def _visualize_samples(self):
        """Visualize sample images from dataset"""
        print(" Visualizing sample images...\n")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.x_train[i].reshape(28, 28), cmap='gray')
            ax.set_title(f'Label: {self.y_train[i]}', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_sample_images.png', dpi=150, bbox_inches='tight')
        print(f" Saved: {self.output_dir}/01_sample_images.png\n")
        plt.close()
        
    def build_model(self):
        """Build CNN model architecture"""
        print("🏗️  Building Neural Network Architecture...\n")
        
        self.model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(" Model built successfully!\n")
        print(self.model.summary())
        print()
        
        # Save model architecture diagram
        keras.utils.plot_model(
            self.model,
            to_file=f'{self.output_dir}/02_model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )
        print(f" Saved: {self.output_dir}/02_model_architecture.png\n")
        
    def train_model(self, epochs=30, batch_size=128):
        """Train the model with callbacks"""
        print(f" Training Model for {epochs} epochs...\n")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{self.output_dir}/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n Training completed!\n")
        
        # Plot training history
        self._plot_training_history()
        
    def _plot_training_history(self):
        """Plot training and validation metrics"""
        print(" Plotting training history...\n")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_training_history.png', dpi=150, bbox_inches='tight')
        print(f" Saved: {self.output_dir}/03_training_history.png\n")
        plt.close()
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print(" Evaluating Model Performance...\n")
        
        # Get predictions
        predictions_proba = self.model.predict(self.x_test, verbose=0)
        self.predictions = np.argmax(predictions_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print("=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print("=" * 80)
        print()
        
        # Classification report
        print("📊 Detailed Classification Report:\n")
        report = classification_report(
            self.y_test, 
            self.predictions,
            target_names=[str(i) for i in range(10)],
            digits=4
        )
        print(report)
        
        # Save classification report
        report_dict = classification_report(
            self.y_test, 
            self.predictions,
            target_names=[str(i) for i in range(10)],
            output_dict=True
        )
        df_report = pd.DataFrame(report_dict).transpose()
        df_report.to_csv(f'{self.output_dir}/04_classification_report.csv')
        print(f" Saved: {self.output_dir}/04_classification_report.csv\n")
        
        # Plot confusion matrix
        self._plot_confusion_matrix()
        
        # Plot per-class metrics
        self._plot_per_class_metrics()
        
        # Plot ROC curves
        self._plot_roc_curves(predictions_proba)
        
        # Show prediction examples
        self._visualize_predictions()
        
        # Show misclassified examples
        self._visualize_misclassified()
        
    def _plot_confusion_matrix(self):
        """Plot confusion matrix"""
        print(" Generating confusion matrix...\n")
        
        cm = confusion_matrix(self.y_test, self.predictions)
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f" Saved: {self.output_dir}/05_confusion_matrix.png\n")
        plt.close()
        
    def _plot_per_class_metrics(self):
        """Plot per-class precision, recall, and F1-score"""
        print(" Plotting per-class metrics...\n")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.predictions
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(10)
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(10)])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.9, 1.0])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_per_class_metrics.png', dpi=150, bbox_inches='tight')
        print(f" Saved: {self.output_dir}/06_per_class_metrics.png\n")
        plt.close()
        
    def _plot_roc_curves(self, predictions_proba):
        """Plot ROC curves for each class"""
        print(" Generating ROC curves...\n")
        
        # Binarize the output
        y_test_bin = label_binarize(self.y_test, classes=range(10))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(10):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], predictions_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'Digit {i} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-class', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_roc_curves.png', dpi=150, bbox_inches='tight')
        print(f" Saved: {self.output_dir}/07_roc_curves.png\n")
        plt.close()
        
    def _visualize_predictions(self):
        """Visualize sample predictions"""
        print(" Visualizing sample predictions...\n")
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.x_test[i].reshape(28, 28), cmap='gray')
            true_label = self.y_test[i]
            pred_label = self.predictions[i]
            
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label} | Pred: {pred_label}', 
                        fontsize=11, color=color, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_sample_predictions.png', dpi=150, bbox_inches='tight')
        print(f" Saved: {self.output_dir}/08_sample_predictions.png\n")
        plt.close()
        
    def _visualize_misclassified(self):
        """Visualize misclassified examples"""
        print(" Visualizing misclassified examples...\n")
        
        # Find misclassified indices
        misclassified = np.where(self.predictions != self.y_test)[0]
        
        if len(misclassified) == 0:
            print(" No misclassified examples! Perfect accuracy!\n")
            return
        
        # Select random misclassified samples
        num_samples = min(15, len(misclassified))
        sample_indices = np.random.choice(misclassified, num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold', color='red')
        
        for i, ax in enumerate(axes.flat):
            if i < len(sample_indices):
                idx = sample_indices[i]
                ax.imshow(self.x_test[idx].reshape(28, 28), cmap='gray')
                ax.set_title(f'True: {self.y_test[idx]} | Pred: {self.predictions[idx]}',
                           fontsize=11, color='red', fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_misclassified.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/09_misclassified.png\n")
        plt.close()
        
    def save_model(self):
        """Save the trained model"""
        model_path = f'{self.output_dir}/mnist_model_final.h5'
        self.model.save(model_path)
        print(f" Model saved: {model_path}\n")
        
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(" Generating summary report...\n")
        
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        report = f"""
{'=' * 80}
MNIST DIGIT CLASSIFICATION - FINAL SUMMARY REPORT
{'=' * 80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
- Training Samples: {len(self.y_train)}
- Validation Samples: {len(self.y_val)}
- Test Samples: {len(self.y_test)}
- Input Shape: (28, 28, 1)
- Number of Classes: 10

MODEL ARCHITECTURE:
- Total Parameters: {self.model.count_params():,}
- Trainable Parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}
- Layers: {len(self.model.layers)}

TRAINING CONFIGURATION:
- Optimizer: Adam (lr=0.001)
- Loss Function: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Epochs Trained: {len(self.history.history['accuracy'])}

FINAL PERFORMANCE:
- Test Loss: {test_loss:.4f}
- Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)

TRAINING HISTORY:
- Best Training Accuracy: {max(self.history.history['accuracy']):.4f}
- Best Validation Accuracy: {max(self.history.history['val_accuracy']):.4f}
- Final Training Loss: {self.history.history['loss'][-1]:.4f}
- Final Validation Loss: {self.history.history['val_loss'][-1]:.4f}

FILES GENERATED:
1. 01_sample_images.png - Sample MNIST digits
2. 02_model_architecture.png - Model architecture diagram
3. 03_training_history.png - Training curves
4. 04_classification_report.csv - Detailed metrics
5. 05_confusion_matrix.png - Confusion matrices
6. 06_per_class_metrics.png - Per-class performance
7. 07_roc_curves.png - ROC curves for each digit
8. 08_sample_predictions.png - Sample predictions
9. 09_misclassified.png - Misclassified examples
10. mnist_model_final.h5 - Saved model
11. summary_report.txt - This file

{'=' * 80}
Built with: TensorFlow {tf.__version__} | Keras {keras.__version__}
Author: AI Coding Assignment 2026
{'=' * 80}
"""
        
        # Save report
        with open(f'{self.output_dir}/summary_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print(f" Saved: {self.output_dir}/summary_report.txt\n")


def main():
    """Main execution function"""
    print("\n Starting MNIST Classification Pipeline...\n")
    
    # Initialize classifier
    classifier = MNISTClassifier()
    
    # Step 1: Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Step 2: Build model
    classifier.build_model()
    
    # Step 3: Train model
    classifier.train_model(epochs=30, batch_size=128)
    
    # Step 4: Evaluate model
    classifier.evaluate_model()
    
    # Step 5: Save model
    classifier.save_model()
    
    # Step 6: Generate summary report
    classifier.generate_summary_report()
    
    print("\n" + "=" * 80)
    print(" MNIST CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n All outputs saved in: {classifier.output_dir}/")
    print("\n Assignment completed with full metrics and visualizations!\n")


if __name__ == "__main__":
    main()
