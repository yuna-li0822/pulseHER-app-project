# =============================================
# Session DataFrame Generator
# Comprehensive feature extraction for ML models
# =============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import logging

# Firebase integration (optional - fallback to local data)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[INFO] Firebase not available, using local data only")

class PPGMorphologyAnalyzer:
    """Advanced PPG morphology feature extraction"""
    
    @staticmethod
    def calculate_stiffness_index(ppg_signal: List[float], sampling_rate: int = 30) -> float:
        """
        Stiffness Index (SI) = Height / Rise Time
        Higher values indicate stiffer arteries
        """
        try:
            signal_array = np.array(ppg_signal)
            
            # Find systolic peak
            peak_idx = np.argmax(signal_array)
            peak_amplitude = signal_array[peak_idx]
            
            # Find diastolic minimum (start of pulse)
            start_idx = 0
            for i in range(peak_idx):
                if signal_array[i] == np.min(signal_array[:peak_idx]):
                    start_idx = i
                    break
            
            # Calculate rise time (seconds)
            rise_time = (peak_idx - start_idx) / sampling_rate
            
            if rise_time > 0:
                si = peak_amplitude / rise_time
                return round(si, 2)
            return 0.0
            
        except Exception as e:
            print(f"[WARN] SI calculation error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_reflection_index(ppg_signal: List[float]) -> float:
        """
        Reflection Index (RI) = Dicrotic Peak / Systolic Peak
        Indicates wave reflection from periphery
        """
        try:
            signal_array = np.array(ppg_signal)
            
            # Find systolic peak
            systolic_peak = np.max(signal_array)
            systolic_idx = np.argmax(signal_array)
            
            # Find dicrotic notch and peak (after systolic peak)
            post_systolic = signal_array[systolic_idx:]
            if len(post_systolic) > 10:
                # Look for local minimum (dicrotic notch) then maximum (dicrotic peak)
                dicrotic_peak = np.max(post_systolic[5:])  # Skip immediate post-systolic area
                
                if systolic_peak > 0:
                    ri = (dicrotic_peak / systolic_peak) * 100
                    return round(ri, 2)
            
            return 0.0
            
        except Exception as e:
            print(f"[WARN] RI calculation error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_dicrotic_notch_amplitude_ratio(ppg_signal: List[float]) -> float:
        """
        Dicrotic Notch Amplitude Ratio (DNAR)
        Ratio of notch depth to pulse amplitude
        """
        try:
            signal_array = np.array(ppg_signal)
            
            systolic_peak = np.max(signal_array)
            systolic_idx = np.argmax(signal_array)
            baseline = np.min(signal_array)
            
            # Find dicrotic notch (minimum after systolic peak)
            post_systolic = signal_array[systolic_idx:]
            if len(post_systolic) > 5:
                notch_value = np.min(post_systolic[2:])  # Skip immediate post-systolic
                
                pulse_amplitude = systolic_peak - baseline
                notch_depth = systolic_peak - notch_value
                
                if pulse_amplitude > 0:
                    dnar = (notch_depth / pulse_amplitude) * 100
                    return round(dnar, 2)
            
            return 0.0
            
        except Exception as e:
            print(f"[WARN] DNAR calculation error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_pulse_timing(ppg_signal: List[float], sampling_rate: int = 30) -> Dict[str, float]:
        """
        Calculate rise time, fall time, and pulse width
        """
        try:
            signal_array = np.array(ppg_signal)
            
            # Find key points
            peak_idx = np.argmax(signal_array)
            baseline = np.min(signal_array)
            peak_amplitude = signal_array[peak_idx]
            
            # Find 10% and 90% points for rise time
            amplitude_10 = baseline + 0.1 * (peak_amplitude - baseline)
            amplitude_90 = baseline + 0.9 * (peak_amplitude - baseline)
            
            # Rise time calculation
            rise_start_idx = 0
            rise_end_idx = peak_idx
            
            for i in range(peak_idx):
                if signal_array[i] >= amplitude_10:
                    rise_start_idx = i
                    break
            
            for i in range(peak_idx):
                if signal_array[i] >= amplitude_90:
                    rise_end_idx = i
                    break
            
            rise_time = (rise_end_idx - rise_start_idx) / sampling_rate
            
            # Fall time (90% to 10% after peak)
            fall_start_idx = peak_idx
            fall_end_idx = len(signal_array) - 1
            
            for i in range(peak_idx, len(signal_array)):
                if signal_array[i] <= amplitude_10:
                    fall_end_idx = i
                    break
            
            fall_time = (fall_end_idx - fall_start_idx) / sampling_rate
            
            # Pulse width (10% to 10%)
            pulse_width = (fall_end_idx - rise_start_idx) / sampling_rate
            
            return {
                'rise_time': round(rise_time, 3),
                'fall_time': round(fall_time, 3),
                'pulse_width': round(pulse_width, 3),
                'pulse_amplitude': round(peak_amplitude - baseline, 2)
            }
            
        except Exception as e:
            print(f"[WARN] Pulse timing calculation error: {e}")
            return {
                'rise_time': 0.0,
                'fall_time': 0.0,
                'pulse_width': 0.0,
                'pulse_amplitude': 0.0
            }

class SignalQualityAnalyzer:
    """Signal quality assessment for PPG data"""
    
    @staticmethod
    def calculate_signal_quality_metrics(ppg_signal: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive signal quality metrics
        """
        try:
            signal_array = np.array(ppg_signal)
            
            # Signal-to-Noise Ratio (SNR)
            signal_power = np.var(signal_array)
            noise_estimate = np.var(np.diff(signal_array))  # High-frequency noise
            snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 0
            
            # Baseline deviation
            baseline_trend = np.polyfit(range(len(signal_array)), signal_array, 1)[0]
            baseline_deviation = abs(baseline_trend)
            
            # Artifact percentage (based on outliers)
            q25, q75 = np.percentile(signal_array, [25, 75])
            iqr = q75 - q25
            outlier_threshold = 1.5 * iqr
            outliers = np.sum((signal_array < (q25 - outlier_threshold)) | 
                            (signal_array > (q75 + outlier_threshold)))
            artifact_percentage = (outliers / len(signal_array)) * 100
            
            # Signal stability (coefficient of variation)
            signal_stability = (np.std(signal_array) / np.mean(signal_array)) * 100 if np.mean(signal_array) > 0 else 0
            
            return {
                'snr': round(snr, 2),
                'baseline_deviation': round(baseline_deviation, 4),
                'artifact_percentage': round(artifact_percentage, 2),
                'signal_stability': round(signal_stability, 2)
            }
            
        except Exception as e:
            print(f"[WARN] Signal quality calculation error: {e}")
            return {
                'snr': 0.0,
                'baseline_deviation': 0.0,
                'artifact_percentage': 0.0,
                'signal_stability': 0.0
            }

class SessionDataFrameGenerator:
    """
    Generate comprehensive pandas DataFrames for ML model training
    Each row = one session, columns = all extracted features
    """
    
    def __init__(self, firebase_config: Optional[Dict] = None):
        self.firebase_config = firebase_config
        self.morphology_analyzer = PPGMorphologyAnalyzer()
        self.quality_analyzer = SignalQualityAnalyzer()
        
        # Initialize Firebase if available
        if FIREBASE_AVAILABLE and firebase_config:
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(firebase_config)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.firebase_enabled = True
                print("[OK] Firebase connection established for DataFrame generation")
            except Exception as e:
                print(f"[WARN] Firebase initialization failed: {e}")
                self.firebase_enabled = False
        else:
            self.firebase_enabled = False
    
    def load_session_data(self, user_id: str = None, days_back: int = 30) -> List[Dict]:
        """
        Load session data from Firebase and/or local storage
        """
        sessions = []
        
        # Load from Firebase
        if self.firebase_enabled:
            try:
                collection_ref = self.db.collection('ppg_sessions')
                query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)
                
                if user_id:
                    query = query.where('user_id', '==', user_id)
                
                # Limit to recent sessions
                cutoff_date = datetime.now() - timedelta(days=days_back)
                query = query.where('timestamp', '>=', cutoff_date)
                
                docs = query.stream()
                
                for doc in docs:
                    session_data = doc.to_dict()
                    session_data['session_id'] = doc.id
                    sessions.append(session_data)
                    
                print(f"[INFO] Loaded {len(sessions)} sessions from Firebase")
                
            except Exception as e:
                print(f"[WARN] Firebase session loading failed: {e}")
        
        # Load from local longitudinal tracking
        try:
            from longitudinal_tracking import get_longitudinal_manager
            manager = get_longitudinal_manager()
            
            if manager:
                user_ids = [user_id] if user_id else manager.get_all_user_ids()
                
                for uid in user_ids:
                    user_sessions = manager.get_user_sessions(uid)
                    for session in user_sessions:
                        # Convert to consistent format
                        local_session = {
                            'session_id': session.session_id,
                            'user_id': uid,
                            'timestamp': session.timestamp,
                            'ppg_data': getattr(session, 'raw_ppg_data', []),
                            'metrics': session.calculated_metrics,
                            'session_type': session.session_type
                        }
                        sessions.append(local_session)
                
                print(f"[INFO] Loaded {len(sessions)} additional sessions from local tracking")
                
        except Exception as e:
            print(f"[WARN] Local session loading failed: {e}")
        
        return sessions
    
    def extract_session_features(self, session_data: Dict) -> Dict[str, float]:
        """
        Extract comprehensive features from a single session
        """
        features = {}
        
        # Session metadata
        features['session_id'] = session_data.get('session_id', '')
        features['user_id'] = session_data.get('user_id', '')
        features['timestamp'] = session_data.get('timestamp', datetime.now().isoformat())
        
        # Get PPG data
        ppg_data = session_data.get('ppg_data', [])
        metrics = session_data.get('metrics', {})
        
        if ppg_data and len(ppg_data) > 10:
            # PPG Morphology Features
            features['stiffness_index'] = self.morphology_analyzer.calculate_stiffness_index(ppg_data)
            features['reflection_index'] = self.morphology_analyzer.calculate_reflection_index(ppg_data)
            features['dicrotic_notch_amplitude_ratio'] = self.morphology_analyzer.calculate_dicrotic_notch_amplitude_ratio(ppg_data)
            
            # Pulse timing features
            timing_features = self.morphology_analyzer.calculate_pulse_timing(ppg_data)
            features.update(timing_features)
            
            # Signal quality features
            quality_features = self.quality_analyzer.calculate_signal_quality_metrics(ppg_data)
            features.update(quality_features)
            
        else:
            # Default values if no PPG data
            features.update({
                'stiffness_index': 0.0,
                'reflection_index': 0.0,
                'dicrotic_notch_amplitude_ratio': 0.0,
                'rise_time': 0.0,
                'fall_time': 0.0,
                'pulse_width': 0.0,
                'pulse_amplitude': 0.0,
                'snr': 0.0,
                'baseline_deviation': 0.0,
                'artifact_percentage': 0.0,
                'signal_stability': 0.0
            })
        
        # HRV Metrics (from existing calculations)
        features['hr'] = metrics.get('HR', 0.0)
        features['rmssd'] = metrics.get('RMSSD', 0.0)
        features['sdnn'] = metrics.get('SDNN', 0.0)
        features['pnn50'] = metrics.get('pNN50', 0.0)
        features['lf'] = metrics.get('LF', 0.0)
        features['hf'] = metrics.get('HF', 0.0)
        features['lf_hf_ratio'] = metrics.get('LF/HF', 0.0)
        features['sd1'] = metrics.get('SD1', 0.0)
        features['sd2'] = metrics.get('SD2', 0.0)
        
        # User Info (static per user, but cycle phase changes)
        user_info = session_data.get('user_info', {})
        features['age'] = user_info.get('age', 0)
        features['bmi'] = user_info.get('bmi', 0.0)
        features['menstrual_cycle_phase'] = user_info.get('cycle_phase', 'unknown')
        features['contraceptive_use'] = 1 if user_info.get('contraceptive_use', False) else 0
        features['pregnancy_history'] = user_info.get('pregnancy_history', 0)
        
        return features
    
    def generate_session_dataframe(self, user_id: str = None, days_back: int = 30) -> pd.DataFrame:
        """
        Generate comprehensive pandas DataFrame with all session features
        Each row = one session, columns = all extracted features
        """
        print(f"🔄 Generating session DataFrame for {'user ' + user_id if user_id else 'all users'}")
        
        # Load session data
        sessions = self.load_session_data(user_id, days_back)
        
        if not sessions:
            print("[WARN] No sessions found, returning empty DataFrame")
            return pd.DataFrame()
        
        # Extract features for each session
        feature_list = []
        for session in sessions:
            try:
                features = self.extract_session_features(session)
                feature_list.append(features)
            except Exception as e:
                print(f"[WARN] Feature extraction failed for session {session.get('session_id', 'unknown')}: {e}")
        
        if not feature_list:
            print("[WARN] No features extracted, returning empty DataFrame")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(feature_list)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp (newest first)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        print(f"✅ Generated DataFrame: {len(df)} sessions × {len(df.columns)} features")
        print(f"📊 Feature columns: {list(df.columns)}")
        
        return df
    
    def save_dataframe(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save DataFrame to CSV file
        """
        if filename is None:
            filename = f"ppg_session_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = f"user_data/{filename}"
        
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            df.to_csv(filepath, index=False)
            print(f"✅ DataFrame saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to save DataFrame: {e}")
            return ""

# Convenience function for easy usage
def generate_ml_ready_dataframe(user_id: str = None, days_back: int = 30, save: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate ML-ready DataFrame
    
    Args:
        user_id: Specific user ID (None for all users)
        days_back: Number of days to look back for sessions
        save: Whether to save DataFrame to CSV
    
    Returns:
        pandas.DataFrame with all session features
    """
    generator = SessionDataFrameGenerator()
    df = generator.generate_session_dataframe(user_id, days_back)
    
    if save and not df.empty:
        generator.save_dataframe(df)
    
    return df

if __name__ == "__main__":
    # Test the system
    print("🧪 Testing Session DataFrame Generator...")
    
    # Generate DataFrame for all users
    df = generate_ml_ready_dataframe(days_back=7, save=True)
    
    if not df.empty:
        print(f"\n📊 DataFrame Summary:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("No data available for testing")