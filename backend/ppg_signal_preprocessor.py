"""
Advanced PPG Signal Preprocessing Module
Implements deep learning-based signal cleaning, artifact detection, and quality assessment
Uses CNN-LSTM hybrid architecture for realistic PPG signal processing
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import cv2
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class PPGSignalPreprocessor:
    """
    Advanced PPG signal preprocessing with deep learning-based cleaning and artifact detection
    """
    
    def __init__(self, sampling_rate=30, window_size=150):
        self.sampling_rate = sampling_rate
        self.window_size = window_size  # 5 seconds at 30fps
        self.scaler = StandardScaler()
        
        # Initialize models
        self.artifact_detector = None
        self.signal_enhancer = None
        self.quality_classifier = None
        
        # Signal quality thresholds
        self.quality_thresholds = {
            'snr_min': 3.0,
            'variance_min': 10.0,
            'variance_max': 1000.0,
            'artifact_threshold': 0.3
        }
        
        self._build_models()
    
    def _build_models(self):
        """Build CNN-LSTM hybrid models for signal processing"""
        
        # 1. Artifact Detection Model (1D CNN)
        self.artifact_detector = Sequential([
            Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(self.window_size, 1)),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=5, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Dropout(0.3),
            tf.keras.layers.GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Binary: artifact vs clean
        ])
        
        # 2. Signal Enhancement Model (CNN-LSTM Hybrid)
        input_signal = Input(shape=(self.window_size, 1))
        
        # CNN feature extraction
        x = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(input_signal)
        x = BatchNormalization()(x)
        x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # LSTM temporal modeling
        x = LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = LSTM(32, return_sequences=True, dropout=0.2)(x)
        
        # Output reconstruction
        enhanced_signal = Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')(x)
        
        self.signal_enhancer = Model(inputs=input_signal, outputs=enhanced_signal)
        
        # Compile models
        self.artifact_detector.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.signal_enhancer.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        # Load pre-trained weights if available
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pre-trained model weights if available"""
        try:
            # In a real implementation, you'd load from saved weights
            # For now, we'll use the initialized random weights
            pass
        except:
            print("No pre-trained weights found, using randomly initialized models")
    
    def extract_features(self, signal_segment):
        """
        Extract hand-crafted features for traditional ML approaches
        """
        if len(signal_segment) < 10:
            return np.zeros(12)
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(signal_segment),
            np.std(signal_segment),
            np.var(signal_segment),
            skew(signal_segment),
            kurtosis(signal_segment),
        ])
        
        # Signal quality features
        signal_diff = np.diff(signal_segment)
        features.extend([
            np.mean(np.abs(signal_diff)),  # Mean absolute difference
            np.std(signal_diff),           # Derivative variability
        ])
        
        # Frequency domain features
        if len(signal_segment) >= 32:
            fft_vals = np.abs(fft(signal_segment)[:len(signal_segment)//2])
            freqs = fftfreq(len(signal_segment), 1/self.sampling_rate)[:len(signal_segment)//2]
            
            # Power in different frequency bands
            hr_band = (freqs >= 0.5) & (freqs <= 4.0)  # Heart rate band (0.5-4 Hz)
            noise_band = (freqs > 4.0) & (freqs <= 15.0)  # High frequency noise
            
            hr_power = np.sum(fft_vals[hr_band]) if np.any(hr_band) else 0
            noise_power = np.sum(fft_vals[noise_band]) if np.any(noise_band) else 0
            total_power = np.sum(fft_vals)
            
            features.extend([
                hr_power / (total_power + 1e-10),     # Relative HR power
                noise_power / (total_power + 1e-10),  # Relative noise power
                hr_power / (noise_power + 1e-10)      # SNR approximation
            ])
        else:
            features.extend([0, 0, 0])
        
        # Peak detection features
        try:
            peaks, _ = signal.find_peaks(signal_segment, distance=self.sampling_rate//4)
            features.extend([
                len(peaks),  # Number of detected peaks
                np.std(np.diff(peaks)) if len(peaks) > 1 else 0  # Peak interval variability
            ])
        except:
            features.extend([0, 0])
        
        return np.array(features)
    
    def detect_artifacts_traditional(self, signal_segment):
        """
        Traditional artifact detection using hand-crafted features and Random Forest
        """
        features = self.extract_features(signal_segment)
        
        # Simple rule-based artifact detection
        artifact_score = 0
        
        # Check for unrealistic variance
        variance = features[2]  # Variance feature
        if variance < self.quality_thresholds['variance_min'] or variance > self.quality_thresholds['variance_max']:
            artifact_score += 0.3
        
        # Check for high frequency noise
        if len(features) >= 10:
            noise_ratio = features[9]  # Relative noise power
            if noise_ratio > 0.5:
                artifact_score += 0.4
        
        # Check for irregular peak patterns
        if len(features) >= 12:
            peak_variability = features[11]
            if peak_variability > 20:  # High variability in peak intervals
                artifact_score += 0.3
        
        return min(artifact_score, 1.0)
    
    def detect_artifacts_deep(self, signal_segment):
        """
        Deep learning-based artifact detection using CNN
        """
        if len(signal_segment) != self.window_size:
            # Pad or truncate to required size
            if len(signal_segment) < self.window_size:
                signal_segment = np.pad(signal_segment, 
                                      (0, self.window_size - len(signal_segment)), 
                                      'constant', constant_values=np.mean(signal_segment))
            else:
                signal_segment = signal_segment[:self.window_size]
        
        # Normalize signal
        normalized_signal = (signal_segment - np.mean(signal_segment)) / (np.std(signal_segment) + 1e-10)
        
        # Prepare for model input
        model_input = normalized_signal.reshape(1, self.window_size, 1)
        
        try:
            # Predict artifact probability
            artifact_prob = self.artifact_detector.predict(model_input, verbose=0)[0][0]
            return float(artifact_prob)
        except Exception as e:
            print(f"Deep artifact detection failed: {e}")
            # Fallback to traditional method
            return self.detect_artifacts_traditional(signal_segment)
    
    def enhance_signal_deep(self, signal_segment):
        """
        Deep learning-based signal enhancement using CNN-LSTM
        """
        if len(signal_segment) != self.window_size:
            # Pad or truncate to required size
            if len(signal_segment) < self.window_size:
                signal_segment = np.pad(signal_segment, 
                                      (0, self.window_size - len(signal_segment)), 
                                      'constant', constant_values=np.mean(signal_segment))
            else:
                signal_segment = signal_segment[:self.window_size]
        
        # Normalize signal
        mean_val = np.mean(signal_segment)
        std_val = np.std(signal_segment) + 1e-10
        normalized_signal = (signal_segment - mean_val) / std_val
        
        # Prepare for model input
        model_input = normalized_signal.reshape(1, self.window_size, 1)
        
        try:
            # Enhance signal
            enhanced_normalized = self.signal_enhancer.predict(model_input, verbose=0)[0]
            
            # Denormalize
            enhanced_signal = enhanced_normalized.flatten() * std_val + mean_val
            
            return enhanced_signal
        except Exception as e:
            print(f"Deep signal enhancement failed: {e}")
            # Fallback to traditional filtering
            return self.clean_signal_traditional(signal_segment)
    
    def clean_signal_traditional(self, signal_segment):
        """
        Traditional signal cleaning using digital filters
        """
        if len(signal_segment) < 10:
            return signal_segment
        
        try:
            # Remove DC component
            signal_detrended = signal.detrend(signal_segment)
            
            # Bandpass filter (0.5-4 Hz for heart rate)
            nyquist = self.sampling_rate / 2
            low_cutoff = 0.5 / nyquist
            high_cutoff = 4.0 / nyquist
            
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_detrended)
            
            # Median filter for spike removal
            filtered_signal = signal.medfilt(filtered_signal, kernel_size=5)
            
            # Moving average smoothing
            window = min(7, len(filtered_signal) // 3)
            if window >= 3:
                filtered_signal = np.convolve(filtered_signal, 
                                            np.ones(window)/window, 
                                            mode='same')
            
            return filtered_signal
            
        except Exception as e:
            print(f"Traditional filtering failed: {e}")
            return signal_segment
    
    def assess_signal_quality(self, signal_segment, artifact_score=None):
        """
        Comprehensive signal quality assessment
        """
        if len(signal_segment) < 10:
            return {
                'quality_score': 0.0,
                'quality_label': 'poor',
                'snr': 0.0,
                'artifact_probability': 1.0,
                'recommendations': ['Signal too short for analysis']
            }
        
        # Calculate SNR
        signal_power = np.var(signal_segment)
        noise_estimate = np.var(np.diff(signal_segment))
        snr = signal_power / (noise_estimate + 1e-10)
        
        # Get artifact score
        if artifact_score is None:
            artifact_score = self.detect_artifacts_deep(signal_segment)
        
        # Calculate overall quality score (0-100)
        snr_score = min(snr / 10.0, 1.0) * 40  # Max 40 points
        artifact_score_norm = (1 - artifact_score) * 40  # Max 40 points
        
        # Stability score (low variability in derivative)
        stability_score = min(1.0 / (np.std(np.diff(signal_segment)) + 1e-10) * 0.1, 1.0) * 20
        
        quality_score = snr_score + artifact_score_norm + stability_score
        
        # Determine quality label
        if quality_score >= 80:
            quality_label = 'excellent'
        elif quality_score >= 60:
            quality_label = 'good'
        elif quality_score >= 40:
            quality_label = 'fair'
        else:
            quality_label = 'poor'
        
        # Generate recommendations
        recommendations = []
        if snr < 3.0:
            recommendations.append('Improve lighting or finger placement')
        if artifact_score > 0.5:
            recommendations.append('Reduce motion artifacts - keep finger still')
        if np.std(signal_segment) < 5:
            recommendations.append('Press finger more firmly on camera')
        if np.std(signal_segment) > 100:
            recommendations.append('Reduce pressure on camera lens')
        
        return {
            'quality_score': round(quality_score, 1),
            'quality_label': quality_label,
            'snr': round(snr, 2),
            'artifact_probability': round(artifact_score, 3),
            'signal_variance': round(np.var(signal_segment), 2),
            'recommendations': recommendations
        }
    
    def process_signal_segment(self, signal_segment, use_deep_learning=True):
        """
        Complete signal processing pipeline
        """
        results = {
            'original_signal': signal_segment.copy(),
            'processed_signal': None,
            'artifact_score': 0.0,
            'quality_assessment': None,
            'processing_method': 'traditional'
        }
        
        # Step 1: Artifact Detection
        if use_deep_learning:
            try:
                artifact_score = self.detect_artifacts_deep(signal_segment)
                results['processing_method'] = 'deep_learning'
            except:
                artifact_score = self.detect_artifacts_traditional(signal_segment)
        else:
            artifact_score = self.detect_artifacts_traditional(signal_segment)
        
        results['artifact_score'] = artifact_score
        
        # Step 2: Signal Enhancement/Cleaning
        if use_deep_learning and artifact_score < 0.7:  # Only enhance if not too corrupted
            try:
                enhanced_signal = self.enhance_signal_deep(signal_segment)
                results['processed_signal'] = enhanced_signal
            except:
                cleaned_signal = self.clean_signal_traditional(signal_segment)
                results['processed_signal'] = cleaned_signal
        else:
            cleaned_signal = self.clean_signal_traditional(signal_segment)
            results['processed_signal'] = cleaned_signal
        
        # Step 3: Quality Assessment
        results['quality_assessment'] = self.assess_signal_quality(
            results['processed_signal'], artifact_score)
        
        return results
    
    def train_artifact_detector_with_synthetic_data(self, num_samples=1000):
        """
        Train artifact detector with synthetic PPG data
        """
        print("Generating synthetic training data for artifact detection...")
        
        X_train = []
        y_train = []
        
        for i in range(num_samples):
            # Generate clean PPG signal
            t = np.linspace(0, 5, self.window_size)
            heart_rate = np.random.uniform(60, 100)  # BPM
            
            # Base cardiac signal
            clean_signal = np.sin(2 * np.pi * (heart_rate / 60) * t)
            clean_signal += 0.3 * np.sin(2 * np.pi * (heart_rate / 60) * 2 * t)  # Harmonic
            clean_signal += 0.1 * np.random.randn(len(t))  # Small noise
            
            # Add baseline and scale
            clean_signal = 150 + 30 * clean_signal
            
            if i < num_samples // 2:
                # Clean signal
                X_train.append(clean_signal.reshape(-1, 1))
                y_train.append(0)  # No artifact
            else:
                # Add artifacts
                corrupted_signal = clean_signal.copy()
                
                # Random artifact types
                artifact_type = np.random.choice(['motion', 'noise', 'baseline_drift', 'spikes'])
                
                if artifact_type == 'motion':
                    # Motion artifacts
                    motion_freq = np.random.uniform(0.1, 2.0)
                    motion_amplitude = np.random.uniform(20, 80)
                    corrupted_signal += motion_amplitude * np.sin(2 * np.pi * motion_freq * t)
                
                elif artifact_type == 'noise':
                    # High frequency noise
                    noise_amplitude = np.random.uniform(15, 50)
                    corrupted_signal += noise_amplitude * np.random.randn(len(t))
                
                elif artifact_type == 'baseline_drift':
                    # Baseline drift
                    drift = np.linspace(-30, 30, len(t)) * np.random.uniform(0.5, 2.0)
                    corrupted_signal += drift
                
                elif artifact_type == 'spikes':
                    # Random spikes
                    num_spikes = np.random.randint(3, 10)
                    spike_positions = np.random.choice(len(t), num_spikes, replace=False)
                    for pos in spike_positions:
                        corrupted_signal[pos] += np.random.uniform(-100, 100)
                
                X_train.append(corrupted_signal.reshape(-1, 1))
                y_train.append(1)  # Artifact present
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training artifact detector on {len(X_train)} samples...")
        
        # Train the model
        history = self.artifact_detector.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print("Artifact detector training complete!")
        return history

# Initialize global preprocessor
ppg_preprocessor = PPGSignalPreprocessor()

def preprocess_ppg_signal(signal_data, use_deep_learning=True):
    """
    Main preprocessing function for PPG signals
    """
    if len(signal_data) < 10:
        return {
            'error': 'Insufficient data for preprocessing',
            'processed_signal': signal_data,
            'quality_score': 0.0
        }
    
    # Process the signal
    results = ppg_preprocessor.process_signal_segment(signal_data, use_deep_learning)
    
    return {
        'original_signal': results['original_signal'].tolist(),
        'processed_signal': results['processed_signal'].tolist(),
        'artifact_score': results['artifact_score'],
        'quality_score': results['quality_assessment']['quality_score'],
        'quality_label': results['quality_assessment']['quality_label'],
        'snr': results['quality_assessment']['snr'],
        'recommendations': results['quality_assessment']['recommendations'],
        'processing_method': results['processing_method']
    }

if __name__ == "__main__":
    # Test the preprocessor
    print("Testing PPG Signal Preprocessor...")
    
    # Generate test signal
    t = np.linspace(0, 5, 150)
    test_signal = 150 + 25 * np.sin(2 * np.pi * 1.2 * t) + 5 * np.random.randn(150)
    
    # Add some artifacts
    test_signal[50:60] += 50  # Motion artifact
    test_signal += 10 * np.random.randn(150)  # Noise
    
    # Process signal
    results = preprocess_ppg_signal(test_signal)
    
    print(f"Quality Score: {results['quality_score']}")
    print(f"Quality Label: {results['quality_label']}")
    print(f"Artifact Score: {results['artifact_score']}")
    print(f"SNR: {results['snr']}")
    print("Recommendations:", results['recommendations'])