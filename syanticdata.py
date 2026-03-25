"""
Integrated Voice Authentication System
Combines RNN, Random Forest, and Isolation Forest for multi-layered security
"""

import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.rnn_model import RNNVoiceAuthenticator
from models.random_forest_model import RFVoiceAuthenticator
from models.isolation_forest_model import IsolationForestAnomalyDetector
from utils.audio_preprocessing import AudioPreprocessor


class IntegratedVoiceAuthSystem:
    """
    Multi-layered voice authentication system combining three approaches:
    1. RNN for deep temporal feature learning
    2. Random Forest for robust classification
    3. Isolation Forest for anomaly detection
    """
    
    def __init__(self, config=None):
        """
        Initialize integrated authentication system
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary for all components
        """
        if config is None:
            config = self._default_config()
        
        self.config = config
        self.preprocessor = AudioPreprocessor(**config['preprocessing'])
        
        # Initialize models
        self.rnn_model = None
        self.rf_model = None
        self.if_model = None
        
        # Authentication thresholds
        self.rnn_threshold = config.get('rnn_threshold', 0.7)
        self.rf_threshold = config.get('rf_threshold', 0.6)
        self.anomaly_threshold = config.get('anomaly_threshold', 0.0)
        
        # Weights for ensemble decision
        self.weights = config.get('ensemble_weights', {
            'rnn': 0.4,
            'rf': 0.4,
            'if': 0.2
        })
        
        self.is_trained = False
        
    def _default_config(self):
        """
        Return default configuration
        """
        return {
            'preprocessing': {
                'sample_rate': 16000,
                'n_mfcc': 40,
                'n_mels': 128
            },
            'rnn': {
                'lstm_units': [128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            },
            'rf': {
                'n_estimators': 200,
                'max_depth': 30,
                'min_samples_split': 5
            },
            'if': {
                'contamination': 0.1,
                'n_estimators': 100
            },
            'rnn_threshold': 0.7,
            'rf_threshold': 0.6,
            'anomaly_threshold': 0.0,
            'ensemble_weights': {
                'rnn': 0.4,
                'rf': 0.4,
                'if': 0.2
            }
        }
    
    def prepare_features(self, audio_files, max_length=200):
        """
        Extract and prepare features from audio files
        
        Parameters:
        -----------
        audio_files : list
            List of audio file paths
        max_length : int
            Maximum sequence length for padding
            
        Returns:
        --------
        features : dict
            Dictionary containing different feature representations
        """
        mfcc_features = []
        flat_features = []
        
        for audio_file in audio_files:
            # Extract features
            features = self.preprocessor.preprocess_pipeline(audio_file)
            
            # MFCC for RNN (temporal features)
            mfcc = features['mfcc']
            if len(mfcc) > max_length:
                mfcc = mfcc[:max_length]
            else:
                pad_width = ((0, max_length - len(mfcc)), (0, 0))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            
            mfcc_features.append(mfcc)
            
            # Flattened features for RF and IF
            # Aggregate MFCC statistics
            mfcc_mean = np.mean(features['mfcc'], axis=0)
            mfcc_std = np.std(features['mfcc'], axis=0)
            mfcc_max = np.max(features['mfcc'], axis=0)
            mfcc_min = np.min(features['mfcc'], axis=0)
            
            # Mel-spectrogram statistics
            mel_mean = np.mean(features['mel_spectrogram'], axis=1)
            mel_std = np.std(features['mel_spectrogram'], axis=1)
            
            # Combine all features
            flat_feat = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min,
                mel_mean, mel_std
            ])
            
            flat_features.append(flat_feat)
        
        return {
            'mfcc': np.array(mfcc_features),
            'flat': np.array(flat_features)
        }
    
    def train(self, X_train_files, y_train, X_val_files=None, y_val=None,
              train_rnn=True, train_rf=True, train_if=True, epochs=50):
        """
        Train all models in the system
        
        Parameters:
        -----------
        X_train_files : list
            List of training audio file paths
        y_train : np.ndarray
            Training labels
        X_val_files : list, optional
            List of validation audio file paths
        y_val : np.ndarray, optional
            Validation labels
        train_rnn : bool
            Whether to train RNN model
        train_rf : bool
            Whether to train Random Forest model
        train_if : bool
            Whether to train Isolation Forest model
        epochs : int
            Number of epochs for RNN training
        """
        print("=" * 60)
        print("INTEGRATED VOICE AUTHENTICATION SYSTEM - TRAINING")
        print("=" * 60)
        
        # Extract features
        print("\n[1/4] Extracting features from training data...")
        train_features = self.prepare_features(X_train_files)
        
        if X_val_files is not None:
            print("Extracting features from validation data...")
            val_features = self.prepare_features(X_val_files)
        else:
            val_features = None
        
        # Get dimensions
        num_classes = len(np.unique(y_train))
        input_shape = train_features['mfcc'].shape[1:]
        
        # Train RNN
        if train_rnn:
            print(f"\n[2/4] Training RNN Model (LSTM)...")
            self.rnn_model = RNNVoiceAuthenticator(
                input_shape=input_shape,
                num_classes=num_classes,
                **self.config['rnn']
            )
            
            if val_features is not None:
                self.rnn_model.train(
                    train_features['mfcc'], y_train,
                    val_features['mfcc'], y_val,
                    epochs=epochs, batch_size=32
                )
            else:
                self.rnn_model.train(
                    train_features['mfcc'], y_train,
                    epochs=epochs, batch_size=32
                )
        
        # Train Random Forest
        if train_rf:
            print(f"\n[3/4] Training Random Forest Model...")
            self.rf_model = RFVoiceAuthenticator(**self.config['rf'])
            self.rf_model.train(train_features['flat'], y_train)
        
        # Train Isolation Forest (on normal samples only)
        if train_if:
            print(f"\n[4/4] Training Isolation Forest (Anomaly Detector)...")
            self.if_model = IsolationForestAnomalyDetector(**self.config['if'])
            # Train on all samples (assumes training data is mostly normal)
            self.if_model.fit(train_features['flat'])
        
        self.is_trained = True
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
    
    def authenticate(self, audio_file, user_id=None, return_details=False):
        """
        Authenticate a voice sample
        
        Parameters:
        -----------
        audio_file : str
            Path to audio file
        user_id : int, optional
            Expected user ID for verification
        return_details : bool
            Whether to return detailed results
            
        Returns:
        --------
        result : dict
            Authentication result
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")
        
        # Extract features
        features = self.prepare_features([audio_file])
        
        # Get predictions from each model
        results = {}
        
        # RNN prediction
        if self.rnn_model is not None:
            rnn_proba = self.rnn_model.predict(features['mfcc'], return_proba=True)[0]
            rnn_pred = np.argmax(rnn_proba)
            rnn_confidence = np.max(rnn_proba)
            results['rnn'] = {
                'predicted_user': rnn_pred,
                'confidence': float(rnn_confidence),
                'authenticated': rnn_confidence > self.rnn_threshold
            }
        
        # Random Forest prediction
        if self.rf_model is not None:
            rf_proba = self.rf_model.predict(features['flat'], return_proba=True)[0]
            rf_pred = np.argmax(rf_proba)
            rf_confidence = np.max(rf_proba)
            results['rf'] = {
                'predicted_user': rf_pred,
                'confidence': float(rf_confidence),
                'authenticated': rf_confidence > self.rf_threshold
            }
        
        # Isolation Forest anomaly detection
        if self.if_model is not None:
            anomaly_score = self.if_model.decision_function(features['flat'])[0]
            is_normal = anomaly_score > self.anomaly_threshold
            results['if'] = {
                'anomaly_score': float(anomaly_score),
                'is_normal': is_normal,
                'authenticated': is_normal
            }
        
        # Ensemble decision
        auth_score = 0
        if 'rnn' in results:
            auth_score += self.weights['rnn'] * results['rnn']['confidence']
        if 'rf' in results:
            auth_score += self.weights['rf'] * results['rf']['confidence']
        if 'if' in results and results['if']['is_normal']:
            auth_score += self.weights['if']
        
        # Final decision
        final_authenticated = auth_score > 0.5
        
        # Check user ID if provided
        if user_id is not None and final_authenticated:
            if 'rnn' in results:
                final_authenticated = final_authenticated and (results['rnn']['predicted_user'] == user_id)
            if 'rf' in results:
                final_authenticated = final_authenticated and (results['rf']['predicted_user'] == user_id)
        
        result = {
            'authenticated': final_authenticated,
            'authentication_score': float(auth_score),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_details:
            result['details'] = results
            result['user_id'] = user_id
        
        return result
    
    def evaluate_system(self, X_test_files, y_test, include_anomalies=False,
                       anomaly_files=None):
        """
        Evaluate complete system performance
        
        Parameters:
        -----------
        X_test_files : list
            List of test audio file paths
        y_test : np.ndarray
            Test labels
        include_anomalies : bool
            Whether to test anomaly detection
        anomaly_files : list, optional
            List of anomalous audio files
            
        Returns:
        --------
        results : dict
            Comprehensive evaluation results
        """
        print("\n" + "=" * 60)
        print("SYSTEM EVALUATION")
        print("=" * 60)
        
        # Extract features
        print("\nExtracting features...")
        test_features = self.prepare_features(X_test_files)
        
        results = {}
        
        # Evaluate RNN
        if self.rnn_model is not None:
            print("\nEvaluating RNN Model...")
            rnn_results = self.rnn_model.evaluate(
                test_features['mfcc'], y_test
            )
            results['rnn'] = rnn_results
            print(f"RNN Accuracy: {rnn_results['accuracy']:.4f}")
        
        # Evaluate Random Forest
        if self.rf_model is not None:
            print("\nEvaluating Random Forest Model...")
            rf_results = self.rf_model.evaluate(
                test_features['flat'], y_test
            )
            results['rf'] = rf_results
            print(f"Random Forest Accuracy: {rf_results['accuracy']:.4f}")
        
        # Evaluate Isolation Forest
        if self.if_model is not None and include_anomalies and anomaly_files is not None:
            print("\nEvaluating Isolation Forest (Anomaly Detection)...")
            
            # Normal samples
            normal_pred = self.if_model.predict(test_features['flat'])
            
            # Anomalous samples
            anomaly_features = self.prepare_features(anomaly_files)
            anomaly_pred = self.if_model.predict(anomaly_features['flat'])
            
            # Combine predictions
            y_if_test = np.concatenate([
                np.ones(len(normal_pred)),
                -np.ones(len(anomaly_pred))
            ])
            y_if_pred = np.concatenate([normal_pred, anomaly_pred])
            
            # Evaluate
            if_results = self.if_model.evaluate(
                np.vstack([test_features['flat'], anomaly_features['flat']]),
                y_if_test
            )
            results['if'] = if_results
            print(f"Detection Rate: {if_results['detection_rate']:.4f}")
            print(f"False Alarm Rate: {if_results['false_alarm_rate']:.4f}")
        
        print("\n" + "=" * 60)
        return results
    
    def save_system(self, directory):
        """
        Save all models and configuration
        
        Parameters:
        -----------
        directory : str
            Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Save models
        if self.rnn_model is not None:
            self.rnn_model.save_model(
                os.path.join(directory, 'rnn_model.h5')
            )
        
        if self.rf_model is not None:
            self.rf_model.save_model(
                os.path.join(directory, 'rf_model.pkl')
            )
        
        if self.if_model is not None:
            self.if_model.save_model(
                os.path.join(directory, 'if_model.pkl')
            )
        
        print(f"System saved to {directory}")
    
    def load_system(self, directory):
        """
        Load all models and configuration
        
        Parameters:
        -----------
        directory : str
            Directory containing saved models
        """
        # Load configuration
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Load models
        rnn_path = os.path.join(directory, 'rnn_model.h5')
        if os.path.exists(rnn_path):
            self.rnn_model = RNNVoiceAuthenticator(
                input_shape=(200, 120),  # Will be overwritten
                num_classes=10  # Will be overwritten
            )
            self.rnn_model.load_model(rnn_path)
        
        rf_path = os.path.join(directory, 'rf_model.pkl')
        if os.path.exists(rf_path):
            self.rf_model = RFVoiceAuthenticator()
            self.rf_model.load_model(rf_path)
        
        if_path = os.path.join(directory, 'if_model.pkl')
        if os.path.exists(if_path):
            self.if_model = IsolationForestAnomalyDetector()
            self.if_model.load_model(if_path)
        
        self.is_trained = True
        print(f"System loaded from {directory}")
