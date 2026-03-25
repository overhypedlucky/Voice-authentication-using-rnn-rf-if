"""
Visualization Module for Voice Authentication System
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class AuthenticationVisualizer:
    """
    Visualization tools for authentication results
    """
    
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None,
                             title='Confusion Matrix', save_name='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        if class_names is not None and len(class_names) <= 20:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=False, cmap='Blues')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training history for RNN model
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', 
                    linewidth=2, marker='o', markersize=4)
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], 
                        label='Validation Accuracy',
                        linewidth=2, marker='s', markersize=4)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training Loss',
                    linewidth=2, marker='o', markersize=4)
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation Loss',
                        linewidth=2, marker='s', markersize=4)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_feature_importance(self, importances, top_n=20,
                               save_name='feature_importance.png'):
        """
        Plot Random Forest feature importance
        """
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), top_importances[::-1], color='steelblue', alpha=0.8)
        plt.yticks(range(top_n), [f'Feature {i}' for i in indices[::-1]])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances (Random Forest)',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_anomaly_scores(self, normal_scores, anomaly_scores,
                           save_name='anomaly_scores.png'):
        """
        Plot anomaly score distribution
        """
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal Samples',
                color='green', edgecolor='black')
        plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomalous Samples',
                color='red', edgecolor='black')
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, 
                   label='Decision Threshold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_model_comparison(self, results_dict, save_name='model_comparison.png'):
        """
        Compare performance across models
        """
        models = list(results_dict.keys())
        accuracies = [results_dict[m]['accuracy'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_authentication_flow(self, save_name='authentication_flow.png'):
        """
        Create system architecture diagram
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Integrated Voice Authentication System Architecture',
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Input layer
        input_box = plt.Rectangle((0.35, 0.85), 0.3, 0.06, 
                                  facecolor='#3498db', edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(0.5, 0.88, 'Voice Input', ha='center', va='center',
               fontsize=12, color='white', fontweight='bold')
        
        # Preprocessing
        preproc_box = plt.Rectangle((0.35, 0.74), 0.3, 0.06,
                                   facecolor='#9b59b6', edgecolor='black', linewidth=2)
        ax.add_patch(preproc_box)
        ax.text(0.5, 0.77, 'Audio Preprocessing', ha='center', va='center',
               fontsize=11, color='white', fontweight='bold')
        
        # Feature extraction
        feat_box = plt.Rectangle((0.35, 0.63), 0.3, 0.06,
                                facecolor='#16a085', edgecolor='black', linewidth=2)
        ax.add_patch(feat_box)
        ax.text(0.5, 0.66, 'Feature Extraction (MFCC, Mel-Spec)',
               ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Three models
        model_y = 0.45
        model_width = 0.22
        model_height = 0.12
        
        # RNN
        rnn_box = plt.Rectangle((0.05, model_y), model_width, model_height,
                               facecolor='#e74c3c', edgecolor='black', linewidth=2)
        ax.add_patch(rnn_box)
        ax.text(0.16, model_y + 0.09, 'RNN Model', ha='center', va='center',
               fontsize=11, color='white', fontweight='bold')
        ax.text(0.16, model_y + 0.06, 'LSTM Architecture', ha='center', va='center',
               fontsize=9, color='white')
        ax.text(0.16, model_y + 0.03, 'Temporal Features', ha='center', va='center',
               fontsize=9, color='white')
        
        # Random Forest
        rf_box = plt.Rectangle((0.39, model_y), model_width, model_height,
                              facecolor='#27ae60', edgecolor='black', linewidth=2)
        ax.add_patch(rf_box)
        ax.text(0.5, model_y + 0.09, 'Random Forest', ha='center', va='center',
               fontsize=11, color='white', fontweight='bold')
        ax.text(0.5, model_y + 0.06, 'Ensemble Learning', ha='center', va='center',
               fontsize=9, color='white')
        ax.text(0.5, model_y + 0.03, 'Statistical Features', ha='center', va='center',
               fontsize=9, color='white')
        
        # Isolation Forest
        if_box = plt.Rectangle((0.73, model_y), model_width, model_height,
                              facecolor='#f39c12', edgecolor='black', linewidth=2)
        ax.add_patch(if_box)
        ax.text(0.84, model_y + 0.09, 'Isolation Forest', ha='center', va='center',
               fontsize=11, color='white', fontweight='bold')
        ax.text(0.84, model_y + 0.06, 'Anomaly Detection', ha='center', va='center',
               fontsize=9, color='white')
        ax.text(0.84, model_y + 0.03, 'Security Layer', ha='center', va='center',
               fontsize=9, color='white')
        
        # Ensemble decision
        ensemble_box = plt.Rectangle((0.35, 0.28), 0.3, 0.06,
                                    facecolor='#8e44ad', edgecolor='black', linewidth=2)
        ax.add_patch(ensemble_box)
        ax.text(0.5, 0.31, 'Ensemble Decision Fusion', ha='center', va='center',
               fontsize=11, color='white', fontweight='bold')
        
        # Output
        output_box = plt.Rectangle((0.35, 0.17), 0.3, 0.06,
                                  facecolor='#2c3e50', edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(0.5, 0.20, 'Authentication Result', ha='center', va='center',
               fontsize=12, color='white', fontweight='bold')
        
        # Security features box
        security_box = plt.Rectangle((0.05, 0.05), 0.9, 0.08,
                                    facecolor='#ecf0f1', edgecolor='#34495e', 
                                    linewidth=2, linestyle='--')
        ax.add_patch(security_box)
        ax.text(0.5, 0.10, 'Security Features: Multi-Factor Authentication | '
               'Spoofing Detection | Real-time Processing',
               ha='center', va='center', fontsize=10, color='#2c3e50',
               style='italic')
        
        # Draw arrows
        arrow_props = dict(arrowstyle='->', lw=2.5, color='#34495e')
        
        ax.annotate('', xy=(0.5, 0.85), xytext=(0.5, 0.80),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, 0.74), xytext=(0.5, 0.69),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.16, 0.63), xytext=(0.16, model_y + model_height),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, 0.63), xytext=(0.5, model_y + model_height),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.84, 0.63), xytext=(0.84, model_y + model_height),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, model_y), xytext=(0.16, model_y),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, model_y), xytext=(0.84, model_y),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, 0.34), xytext=(0.5, 0.28),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, 0.23), xytext=(0.5, 0.17),
                   arrowprops=arrow_props)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
