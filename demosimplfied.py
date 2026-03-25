"""
Simplified demonstration without TensorFlow dependency
Demonstrates Random Forest and Isolation Forest only
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.random_forest_model import RFVoiceAuthenticator
from models.isolation_forest_model import IsolationForestAnomalyDetector
from utils.audio_preprocessing import AudioPreprocessor
from utils.synthetic_data_generator import SyntheticVoiceGenerator
from utils.visualization import AuthenticationVisualizer
from sklearn.model_selection import train_test_split
import json


def generate_synthetic_dataset():
    """Generate synthetic voice dataset"""
    print("\n" + "="*70)
    print("STEP 1: GENERATING SYNTHETIC VOICE DATASET")
    print("="*70)
    
    generator = SyntheticVoiceGenerator()
    file_paths, labels = generator.generate_dataset(
        num_speakers=10,
        samples_per_speaker=20,
        num_anomalies=50,
        output_dir='/home/sandbox/voice_auth_system/data/synthetic_voice_data'
    )
    
    return file_paths, labels


def extract_features(file_paths):
    """Extract features from audio files"""
    print("\nExtracting features from audio files...")
    preprocessor = AudioPreprocessor()
    features_list = []
    
    for i, file_path in enumerate(file_paths):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(file_paths)} files...")
        
        try:
            features = preprocessor.preprocess_pipeline(file_path)
            
            # Aggregate MFCC statistics
            mfcc_mean = np.mean(features['mfcc'], axis=0)
            mfcc_std = np.std(features['mfcc'], axis=0)
            mfcc_max = np.max(features['mfcc'], axis=0)
            mfcc_min = np.min(features['mfcc'], axis=0)
            
            # Mel-spectrogram statistics
            mel_mean = np.mean(features['mel_spectrogram'], axis=1)
            mel_std = np.std(features['mel_spectrogram'], axis=1)
            
            # Combine
            flat_feat = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min,
                mel_mean, mel_std
            ])
            
            features_list.append(flat_feat)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            features_list.append(np.zeros(616))  # Default feature size
    
    print(f"  Feature extraction complete. Shape: {np.array(features_list).shape}")
    return np.array(features_list)


def main():
    """Main demonstration"""
    print("\n" + "="*70)
    print("VOICE AUTHENTICATION SYSTEM - DEMONSTRATION")
    print("Random Forest + Isolation Forest Implementation")
    print("="*70)
    
    # Generate dataset
    file_paths, labels = generate_synthetic_dataset()
    
    # Split dataset
    print("\n" + "="*70)
    print("STEP 2: SPLITTING DATASET")
    print("="*70)
    
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    X_train_files, X_temp_files, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=0.4, random_state=42, stratify=labels
    )
    
    X_val_files, X_test_files, y_val, y_test = train_test_split(
        X_temp_files, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train_files)}")
    print(f"Validation samples: {len(X_val_files)}")
    print(f"Test samples: {len(X_test_files)}")
    
    # Extract features
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING FEATURES")
    print("="*70)
    
    X_train = extract_features(X_train_files)
    X_val = extract_features(X_val_files)
    X_test = extract_features(X_test_files)
    
    # Train Random Forest
    print("\n" + "="*70)
    print("STEP 4: TRAINING RANDOM FOREST MODEL")
    print("="*70)
    
    rf_model = RFVoiceAuthenticator(
        n_estimators=200,
        max_depth=30,
        random_state=42
    )
    rf_model.train(X_train, y_train)
    
    # Evaluate Random Forest
    print("\n" + "="*70)
    print("STEP 5: EVALUATING RANDOM FOREST")
    print("="*70)
    
    rf_results = rf_model.evaluate(X_test, y_test)
    print(f"\nRandom Forest Accuracy: {rf_results['accuracy']:.4f}")
    print(f"Classification Report:")
    for label, metrics in rf_results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  Class {label}: Precision={metrics.get('precision', 0):.3f}, "
                  f"Recall={metrics.get('recall', 0):.3f}, "
                  f"F1-Score={metrics.get('f1-score', 0):.3f}")
    
    # Train Isolation Forest
    print("\n" + "="*70)
    print("STEP 6: TRAINING ISOLATION FOREST (ANOMALY DETECTOR)")
    print("="*70)
    
    if_model = IsolationForestAnomalyDetector(
        contamination=0.1,
        n_estimators=100,
        random_state=42
    )
    if_model.fit(X_train)
    
    # Evaluate Isolation Forest
    print("\n" + "="*70)
    print("STEP 7: EVALUATING ANOMALY DETECTION")
    print("="*70)
    
    # Get anomaly samples
    anomaly_dir = '/home/sandbox/voice_auth_system/data/synthetic_voice_data/anomaly'
    anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)
                     if f.endswith('.wav')]
    X_anomaly = extract_features(anomaly_files)
    
    # Predict on normal and anomalous samples
    normal_scores = if_model.decision_function(X_test)
    anomaly_scores = if_model.decision_function(X_anomaly)
    
    # Combine for evaluation
    X_combined = np.vstack([X_test, X_anomaly])
    y_combined = np.concatenate([
        np.ones(len(X_test)),  # Normal = 1
        -np.ones(len(X_anomaly))  # Anomaly = -1
    ])
    
    if_results = if_model.evaluate(X_combined, y_combined)
    
    print(f"\nDetection Rate: {if_results['detection_rate']:.4f}")
    print(f"False Alarm Rate: {if_results['false_alarm_rate']:.4f}")
    if if_results['roc_auc'] is not None:
        print(f"ROC-AUC Score: {if_results['roc_auc']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = AuthenticationVisualizer(
        output_dir='/home/sandbox/voice_auth_system/visualizations'
    )
    
    # System architecture
    print("Creating system architecture diagram...")
    arch_path = visualizer.plot_authentication_flow()
    print(f"  Saved: {arch_path}")
    
    # Confusion matrix
    print("Creating confusion matrix...")
    cm_path = visualizer.plot_confusion_matrix(
        y_test, rf_results['predictions'],
        title='Random Forest - Confusion Matrix',
        save_name='rf_confusion_matrix.png'
    )
    print(f"  Saved: {cm_path}")
    
    # Feature importance
    print("Plotting feature importance...")
    fi_path = visualizer.plot_feature_importance(
        rf_model.feature_importance_,
        top_n=20,
        save_name='rf_feature_importance.png'
    )
    print(f"  Saved: {fi_path}")
    
    # Anomaly scores
    print("Plotting anomaly score distribution...")
    as_path = visualizer.plot_anomaly_scores(
        normal_scores, anomaly_scores,
        save_name='anomaly_score_distribution.png'
    )
    print(f"  Saved: {as_path}")
    
    # Model comparison
    print("Creating model comparison chart...")
    comparison_results = {
        'Random Forest': {'accuracy': rf_results['accuracy']}
    }
    comp_path = visualizer.plot_model_comparison(
        comparison_results,
        save_name='model_comparison.png'
    )
    print(f"  Saved: {comp_path}")
    
    # Save results
    print("\n" + "="*70)
    print("STEP 9: SAVING RESULTS")
    print("="*70)
    
    results = {
        'random_forest': {
            'accuracy': float(rf_results['accuracy']),
            'classification_report': {
                k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                    for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in rf_results['classification_report'].items()
            }
        },
        'isolation_forest': {
            'detection_rate': float(if_results['detection_rate']),
            'false_alarm_rate': float(if_results['false_alarm_rate']),
            'roc_auc': float(if_results['roc_auc']) if if_results['roc_auc'] else None
        }
    }
    
    output_path = '/home/sandbox/voice_auth_system/evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {output_path}")
    
    # Save models
    print("\nSaving trained models...")
    os.makedirs('/home/sandbox/voice_auth_system/saved_models', exist_ok=True)
    rf_model.save_model('/home/sandbox/voice_auth_system/saved_models/rf_model.pkl')
    if_model.save_model('/home/sandbox/voice_auth_system/saved_models/if_model.pkl')
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Outputs:")
    print("  - Trained models: /home/sandbox/voice_auth_system/saved_models/")
    print("  - Visualizations: /home/sandbox/voice_auth_system/visualizations/")
    print("  - Results: /home/sandbox/voice_auth_system/evaluation_results.json")
    print("  - Dataset: /home/sandbox/voice_auth_system/data/synthetic_voice_data/")
    print("\nKey Results:")
    print(f"  - Random Forest Accuracy: {rf_results['accuracy']:.4f}")
    print(f"  - Anomaly Detection Rate: {if_results['detection_rate']:.4f}")
    print(f"  - False Alarm Rate: {if_results['false_alarm_rate']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
