"""
Complete Demonstration of Integrated Voice Authentication System
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from integrated_system import IntegratedVoiceAuthSystem
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


def split_dataset(file_paths, labels):
    """Split dataset into train, validation, and test sets"""
    print("\n" + "="*70)
    print("STEP 2: SPLITTING DATASET")
    print("="*70)
    
    # Convert to numpy arrays
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    # Split: 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=0.4, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_system(X_train, y_train, X_val, y_val):
    """Train the integrated authentication system"""
    print("\n" + "="*70)
    print("STEP 3: TRAINING INTEGRATED AUTHENTICATION SYSTEM")
    print("="*70)
    
    # Initialize system
    system = IntegratedVoiceAuthSystem()
    
    # Train all models
    system.train(
        X_train_files=list(X_train),
        y_train=y_train,
        X_val_files=list(X_val),
        y_val=y_val,
        train_rnn=True,
        train_rf=True,
        train_if=True,
        epochs=30
    )
    
    return system


def evaluate_system(system, X_test, y_test):
    """Evaluate system performance"""
    print("\n" + "="*70)
    print("STEP 4: EVALUATING SYSTEM PERFORMANCE")
    print("="*70)
    
    # Get anomaly files
    anomaly_dir = '/home/sandbox/voice_auth_system/data/synthetic_voice_data/anomaly'
    anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)
                     if f.endswith('.wav')]
    
    # Evaluate
    results = system.evaluate_system(
        X_test_files=list(X_test),
        y_test=y_test,
        include_anomalies=True,
        anomaly_files=anomaly_files
    )
    
    return results


def visualize_results(system, results, X_test, y_test):
    """Create visualizations"""
    print("\n" + "="*70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = AuthenticationVisualizer(
        output_dir='/home/sandbox/voice_auth_system/visualizations'
    )
    
    # System architecture
    print("Creating system architecture diagram...")
    arch_path = visualizer.plot_authentication_flow()
    print(f"Saved: {arch_path}")
    
    # Training history (RNN)
    if system.rnn_model and system.rnn_model.history:
        print("Plotting RNN training history...")
        hist_path = visualizer.plot_training_history(
            system.rnn_model.history,
            save_name='rnn_training_history.png'
        )
        print(f"Saved: {hist_path}")
    
    # Confusion matrices
    if 'rnn' in results:
        print("Creating RNN confusion matrix...")
        test_features = system.prepare_features(list(X_test))
        rnn_pred = system.rnn_model.predict(test_features['mfcc'])
        cm_path = visualizer.plot_confusion_matrix(
            y_test, rnn_pred,
            title='RNN Model - Confusion Matrix',
            save_name='rnn_confusion_matrix.png'
        )
        print(f"Saved: {cm_path}")
    
    if 'rf' in results:
        print("Creating Random Forest confusion matrix...")
        rf_pred = results['rf']['predictions']
        cm_path = visualizer.plot_confusion_matrix(
            y_test, rf_pred,
            title='Random Forest - Confusion Matrix',
            save_name='rf_confusion_matrix.png'
        )
        print(f"Saved: {cm_path}")
    
    # Feature importance
    if system.rf_model and system.rf_model.feature_importance_ is not None:
        print("Plotting feature importance...")
        fi_path = visualizer.plot_feature_importance(
            system.rf_model.feature_importance_,
            top_n=20,
            save_name='rf_feature_importance.png'
        )
        print(f"Saved: {fi_path}")
    
    # Anomaly scores
    if 'if' in results:
        print("Plotting anomaly score distribution...")
        # Get scores for normal and anomalous samples
        test_features = system.prepare_features(list(X_test))
        normal_scores = system.if_model.decision_function(test_features['flat'])
        
        anomaly_dir = '/home/sandbox/voice_auth_system/data/synthetic_voice_data/anomaly'
        anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)
                        if f.endswith('.wav')]
        anomaly_features = system.prepare_features(anomaly_files)
        anomaly_scores = system.if_model.decision_function(anomaly_features['flat'])
        
        as_path = visualizer.plot_anomaly_scores(
            normal_scores, anomaly_scores,
            save_name='anomaly_score_distribution.png'
        )
        print(f"Saved: {as_path}")
    
    # Model comparison
    print("Creating model comparison chart...")
    comparison_results = {
        'RNN': {'accuracy': results['rnn']['accuracy']},
        'Random Forest': {'accuracy': results['rf']['accuracy']},
    }
    comp_path = visualizer.plot_model_comparison(
        comparison_results,
        save_name='model_comparison.png'
    )
    print(f"Saved: {comp_path}")


def test_individual_authentication(system):
    """Test authentication on individual samples"""
    print("\n" + "="*70)
    print("STEP 6: TESTING INDIVIDUAL AUTHENTICATION")
    print("="*70)
    
    # Test on a few samples
    test_dir = '/home/sandbox/voice_auth_system/data/synthetic_voice_data/normal'
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                  if f.endswith('.wav')][:5]
    
    for i, test_file in enumerate(test_files):
        print(f"\nTest {i+1}: {os.path.basename(test_file)}")
        result = system.authenticate(test_file, return_details=True)
        
        print(f"  Authenticated: {result['authenticated']}")
        print(f"  Authentication Score: {result['authentication_score']:.4f}")
        
        if 'details' in result:
            if 'rnn' in result['details']:
                print(f"  RNN Confidence: {result['details']['rnn']['confidence']:.4f}")
            if 'rf' in result['details']:
                print(f"  RF Confidence: {result['details']['rf']['confidence']:.4f}")
            if 'if' in result['details']:
                print(f"  Anomaly Score: {result['details']['if']['anomaly_score']:.4f}")


def save_results_report(results):
    """Save detailed results to JSON"""
    print("\n" + "="*70)
    print("STEP 7: SAVING RESULTS REPORT")
    print("="*70)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    # Save to JSON
    output_path = '/home/sandbox/voice_auth_system/evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"Results saved to: {output_path}")


def main():
    """Main demonstration function"""
    print("\n" + "="*70)
    print("INTEGRATED VOICE AUTHENTICATION SYSTEM - COMPLETE DEMONSTRATION")
    print("="*70)
    print("This demonstration showcases:")
    print("  1. Synthetic voice data generation")
    print("  2. Multi-algorithm authentication (RNN + Random Forest + Isolation Forest)")
    print("  3. Comprehensive evaluation and visualization")
    print("="*70)
    
    # Step 1: Generate dataset
    file_paths, labels = generate_synthetic_dataset()
    
    # Step 2: Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(file_paths, labels)
    
    # Step 3: Train system
    system = train_system(X_train, y_train, X_val, y_val)
    
    # Step 4: Evaluate system
    results = evaluate_system(system, X_test, y_test)
    
    # Step 5: Visualize results
    visualize_results(system, results, X_test, y_test)
    
    # Step 6: Test individual authentication
    test_individual_authentication(system)
    
    # Step 7: Save results
    save_results_report(results)
    
    # Step 8: Save trained system
    print("\n" + "="*70)
    print("STEP 8: SAVING TRAINED SYSTEM")
    print("="*70)
    system.save_system('/home/sandbox/voice_auth_system/saved_models')
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Outputs:")
    print("  - Trained models: /home/sandbox/voice_auth_system/saved_models/")
    print("  - Visualizations: /home/sandbox/voice_auth_system/visualizations/")
    print("  - Results: /home/sandbox/voice_auth_system/evaluation_results.json")
    print("  - Dataset: /home/sandbox/voice_auth_system/data/synthetic_voice_data/")
    print("="*70)


if __name__ == "__main__":
    main()
