
"""
Audio Preprocessing Module for Voice Authentication System
Handles feature extraction, noise reduction, and audio augmentation
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import wiener
from python_speech_features import mfcc, delta
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Comprehensive audio preprocessing for voice authentication
    """
    
    def __init__(self, sample_rate=16000, n_mfcc=40, n_mels=128):
        """
        Initialize audio preprocessor
        
        Parameters:
        -----------
        sample_rate : int
            Target sampling rate for audio processing
        n_mfcc : int
            Number of MFCC coefficients to extract
        n_mels : int
            Number of mel bands for mel-spectrogram
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        
    def load_audio(self, file_path, duration=None):
        """
        Load audio file with resampling
        
        Parameters:
        -----------
        file_path : str
            Path to audio file
        duration : float, optional
            Maximum duration to load in seconds
            
        Returns:
        --------
        audio : np.ndarray
            Audio time series
        sr : int
            Sampling rate
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                    duration=duration, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def reduce_noise(self, audio):
        """
        Apply Wiener filtering for noise reduction
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
            
        Returns:
        --------
        denoised : np.ndarray
            Denoised audio signal
        """
        # Apply Wiener filter
        denoised = wiener(audio, mysize=5)
        return denoised
    
    def normalize_audio(self, audio):
        """
        Normalize audio amplitude
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
            
        Returns:
        --------
        normalized : np.ndarray
            Normalized audio signal
        """
        if np.max(np.abs(audio)) > 0:
            normalized = audio / np.max(np.abs(audio))
        else:
            normalized = audio
        return normalized
    
    def extract_mfcc(self, audio, sr=None):
        """
        Extract MFCC features with deltas and delta-deltas
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int, optional
            Sampling rate
            
        Returns:
        --------
        features : np.ndarray
            MFCC features with shape (n_frames, n_mfcc * 3)
        """
        if sr is None:
            sr = self.sample_rate
            
        # Extract MFCC
        mfcc_feat = mfcc(audio, samplerate=sr, numcep=self.n_mfcc,
                        nfilt=self.n_mels, nfft=512, 
                        winlen=0.025, winstep=0.01)
        
        # Calculate delta and delta-delta
        mfcc_delta = delta(mfcc_feat, 2)
        mfcc_delta2 = delta(mfcc_delta, 2)
        
        # Concatenate features
        features = np.hstack([mfcc_feat, mfcc_delta, mfcc_delta2])
        
        return features
    
    def extract_mel_spectrogram(self, audio, sr=None):
        """
        Extract mel-spectrogram features
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int, optional
            Sampling rate
            
        Returns:
        --------
        mel_spec : np.ndarray
            Mel-spectrogram with shape (n_mels, n_frames)
        """
        if sr is None:
            sr = self.sample_rate
            
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=self.n_mels,
            n_fft=2048, hop_length=512, fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_spectral_features(self, audio, sr=None):
        """
        Extract additional spectral features
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int, optional
            Sampling rate
            
        Returns:
        --------
        features : dict
            Dictionary containing various spectral features
        """
        if sr is None:
            sr = self.sample_rate
            
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=sr)[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=sr)[0]
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio, sr=sr)
        
        return features
    
    def augment_audio(self, audio, augmentation_type='noise'):
        """
        Apply data augmentation to audio
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        augmentation_type : str
            Type of augmentation ('noise', 'pitch', 'speed', 'time_stretch')
            
        Returns:
        --------
        augmented : np.ndarray
            Augmented audio signal
        """
        if augmentation_type == 'noise':
            # Add Gaussian noise
            noise = np.random.randn(len(audio)) * 0.005
            augmented = audio + noise
            
        elif augmentation_type == 'pitch':
            # Pitch shifting
            augmented = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=2)
            
        elif augmentation_type == 'speed':
            # Time stretching (speed change)
            augmented = librosa.effects.time_stretch(audio, rate=1.1)
            
        elif augmentation_type == 'time_stretch':
            # Time stretching
            augmented = librosa.effects.time_stretch(audio, rate=0.9)
            
        else:
            augmented = audio
            
        return augmented
    
    def preprocess_pipeline(self, file_path, denoise=True, normalize=True):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        file_path : str
            Path to audio file
        denoise : bool
            Whether to apply denoising
        normalize : bool
            Whether to normalize audio
            
        Returns:
        --------
        features : dict
            Dictionary containing all extracted features
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Apply denoising if requested
        if denoise:
            audio = self.reduce_noise(audio)
        
        # Normalize if requested
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Extract all features
        features = {
            'mfcc': self.extract_mfcc(audio, sr),
            'mel_spectrogram': self.extract_mel_spectrogram(audio, sr),
            'spectral_features': self.extract_spectral_features(audio, sr),
            'raw_audio': audio,
            'sample_rate': sr
        }
        
        return features
    
    def pad_sequence(self, sequence, max_length, pad_value=0):
        """
        Pad or truncate sequence to fixed length
        
        Parameters:
        -----------
        sequence : np.ndarray
            Input sequence
        max_length : int
            Target length
        pad_value : float
            Value to use for padding
            
        Returns:
        --------
        padded : np.ndarray
            Padded/truncated sequence
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        elif len(sequence) < max_length:
            pad_width = max_length - len(sequence)
            return np.pad(sequence, (0, pad_width), 
                         mode='constant', constant_values=pad_value)
        return sequence
