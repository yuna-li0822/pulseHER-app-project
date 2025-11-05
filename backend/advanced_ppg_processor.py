"""
Enhanced PPG Processor for PulseHER v4.0
Advanced HRV Metrics & Cycle-Aware Analysis
Based on clinical research specifications
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import cv2
import time
import math
import statistics
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime, timedelta
import logging

class AdvancedPPGProcessor:
    """
    Enhanced PPG processor with clinical-grade HRV metrics
    and cycle-aware analysis for women's health
    """
    
    def __init__(self, sampling_rate: int = 30):
        """Initialize enhanced PPG processor"""
        self.sampling_rate = sampling_rate
        self.buffer_size = 300  # 10 seconds at 30 fps
        self.ppg_buffer = []
        self.timestamps = []
        self.rr_intervals = []
        
        # Quality thresholds
        self.min_rr_ms = 250  # Minimum RR interval (240 BPM max)
        self.max_rr_ms = 2000  # Maximum RR interval (30 BPM min)
        self.artifact_threshold = 0.2  # 20% change threshold
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_rr_intervals(self, rr_ms: List[float]) -> Tuple[List[float], float]:
        """
        Clean RR intervals using clinical-grade artifact detection
        
        Args:
            rr_ms: Raw RR intervals in milliseconds
            
        Returns:
            Tuple of (cleaned_rr_intervals, artifact_percentage)
        """
        if not rr_ms or len(rr_ms) < 2:
            return [], 100.0
        
        # Step 1: Remove physiologically impossible intervals
        physiological_rr = [r for r in rr_ms if self.min_rr_ms <= r <= self.max_rr_ms]
        
        # Step 2: Remove sudden jumps (>20% change)
        cleaned = []
        artifacts = 0
        
        if physiological_rr:
            cleaned.append(physiological_rr[0])
            
            for i in range(1, len(physiological_rr)):
                current_rr = physiological_rr[i]
                prev_rr = cleaned[-1]
                
                # Check for sudden change
                change_ratio = abs(current_rr - prev_rr) / prev_rr
                
                if change_ratio <= self.artifact_threshold:
                    cleaned.append(current_rr)
                else:
                    artifacts += 1
        
        # Calculate artifact percentage
        total_intervals = len(rr_ms)
        artifact_pct = (artifacts + (total_intervals - len(physiological_rr))) / total_intervals * 100
        
        return cleaned, artifact_pct
    
    def compute_time_domain_metrics(self, rr_ms: List[float]) -> Dict[str, float]:
        """
        Compute comprehensive time-domain HRV metrics
        
        Args:
            rr_ms: Cleaned RR intervals in milliseconds
            
        Returns:
            Dictionary of time-domain metrics
        """
        if len(rr_ms) < 5:
            return {}
        
        # Convert to numpy array for efficiency
        rr = np.array(rr_ms)
        
        # Basic statistics
        mean_rr = np.mean(rr)
        mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
        
        # SDNN: Standard deviation of NN intervals
        sdnn = np.std(rr, ddof=1)
        
        # Successive differences
        diff_rr = np.diff(rr)
        
        # RMSSD: Root mean square of successive differences
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        
        # pNN50: Percentage of successive RR intervals that differ by more than 50ms
        nn50_count = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50_count / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        # pNN20: More sensitive measure
        nn20_count = np.sum(np.abs(diff_rr) > 20)
        pnn20 = (nn20_count / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        # Coefficient of variation (CV)
        cv = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        
        return {
            'mean_hr_bpm': mean_hr,
            'mean_rr_ms': mean_rr,
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'pnn50_pct': pnn50,
            'pnn20_pct': pnn20,
            'cv_pct': cv,
            'total_beats': len(rr)
        }
    
    def compute_frequency_domain_metrics(self, rr_ms: List[float]) -> Dict[str, float]:
        """
        Compute frequency-domain HRV metrics (LF/HF analysis)
        
        Args:
            rr_ms: Cleaned RR intervals in milliseconds
            
        Returns:
            Dictionary of frequency-domain metrics
        """
        if len(rr_ms) < 50:  # Need sufficient data for frequency analysis
            return {}
        
        # Interpolate RR intervals to regular sampling
        rr = np.array(rr_ms)
        time_cumsum = np.cumsum(rr) / 1000.0  # Convert to seconds
        
        # Create regular time grid (4 Hz sampling recommended for HRV)
        fs_interp = 4.0
        time_regular = np.arange(0, time_cumsum[-1], 1/fs_interp)
        
        # Interpolate RR series
        rr_interp = np.interp(time_regular, time_cumsum, rr)
        
        # Detrend
        rr_detrend = signal.detrend(rr_interp)
        
        # Apply Hamming window
        windowed = rr_detrend * signal.hamming(len(rr_detrend))
        
        # Compute power spectral density
        freqs, psd = signal.welch(windowed, fs=fs_interp, nperseg=min(256, len(windowed)//2))
        
        # Define frequency bands (Hz)
        vlf_band = (0.0033, 0.04)  # Very Low Frequency
        lf_band = (0.04, 0.15)     # Low Frequency (sympathetic + parasympathetic)
        hf_band = (0.15, 0.4)      # High Frequency (parasympathetic)
        
        # Calculate power in each band
        vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs <= vlf_band[1])])
        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])
        
        # Total power
        total_power = vlf_power + lf_power + hf_power
        
        # Normalized powers
        lf_nu = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        hf_nu = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        
        # LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
            'lf_hf_ratio': lf_hf_ratio
        }
    
    def estimate_cycle_phase(self, last_period_start: str, cycle_length: int = 28) -> Dict[str, Union[str, int]]:
        """
        Estimate menstrual cycle phase for cycle-aware analysis
        
        Args:
            last_period_start: Date string (YYYY-MM-DD)
            cycle_length: Cycle length in days
            
        Returns:
            Dictionary with phase info
        """
        try:
            start_date = datetime.strptime(last_period_start, "%Y-%m-%d")
            today = datetime.now()
            
            days_since_period = (today - start_date).days
            day_in_cycle = (days_since_period % cycle_length) + 1
            
            # Phase estimation
            if 1 <= day_in_cycle <= 5:
                phase = "menstrual"
                phase_color = "#F48FB1"
            elif 6 <= day_in_cycle <= 13:
                phase = "follicular"
                phase_color = "#FCE4EC"
            elif 14 <= day_in_cycle <= 15:
                phase = "ovulation"
                phase_color = "#FFD54F"
            elif 16 <= day_in_cycle <= cycle_length:
                phase = "luteal"
                phase_color = "#BA68C8"
            else:
                phase = "unknown"
                phase_color = "#E0E0E0"
            
            return {
                'day_in_cycle': day_in_cycle,
                'phase': phase,
                'phase_color': phase_color,
                'cycle_length': cycle_length,
                'days_since_period': days_since_period
            }
        
        except Exception as e:
            self.logger.error(f"Cycle phase estimation error: {e}")
            return {
                'day_in_cycle': 1,
                'phase': 'unknown',
                'phase_color': '#E0E0E0',
                'cycle_length': cycle_length,
                'days_since_period': 0
            }
    
    def compute_derived_indices(self, time_metrics: Dict, freq_metrics: Dict, 
                               baseline_metrics: Dict = None) -> Dict[str, float]:
        """
        Compute derived indices (ABI, CVR, CSI)
        
        Args:
            time_metrics: Time-domain HRV metrics
            freq_metrics: Frequency-domain HRV metrics
            baseline_metrics: User's baseline metrics for normalization
            
        Returns:
            Dictionary of derived indices
        """
        # Extract key metrics
        hr = time_metrics.get('mean_hr_bpm', 70)
        rmssd = time_metrics.get('rmssd_ms', 30)
        lf_hf = freq_metrics.get('lf_hf_ratio', 1.0)
        
        # ABI: Autonomic Balance Index
        # Higher values indicate sympathetic dominance
        if baseline_metrics:
            baseline_hr = baseline_metrics.get('mean_hr_bpm', 70)
            baseline_rmssd = baseline_metrics.get('rmssd_ms', 30)
            
            # Z-scores relative to baseline
            z_hr = (hr - baseline_hr) / max(baseline_hr * 0.1, 5)  # Prevent division by very small numbers
            z_rmssd = (rmssd - baseline_rmssd) / max(baseline_rmssd * 0.1, 2)
            
            abi = 50 + 10 * ((z_hr - z_rmssd) / 2)
        else:
            # Population-based normalization (fallback)
            abi = 50 + 10 * ((hr - 70) / 10 - (rmssd - 30) / 15) / 2
        
        # CVR: Cardiovagal Resilience
        # Based on parasympathetic activity (HF power and RMSSD)
        hf_power = freq_metrics.get('hf_power', 100)
        cvr = (rmssd * np.log(hf_power + 1)) / 10 if hf_power > 0 else rmssd / 10
        
        # CSI: Cycle Stability Index (simplified version)
        # Measures consistency with expected patterns
        sdnn = time_metrics.get('sdnn_ms', 40)
        cv = time_metrics.get('cv_pct', 5)
        csi = max(0, 100 - cv)  # Lower coefficient of variation = higher stability
        
        return {
            'abi': np.clip(abi, 0, 100),  # Clip to reasonable range
            'cvr': np.clip(cvr, 0, 100),
            'csi': np.clip(csi, 0, 100),
            'stress_index': np.clip(lf_hf * 10, 0, 100)  # Simple stress indicator
        }
    
    def assess_signal_quality(self, rr_ms: List[float], artifact_pct: float) -> Dict[str, Union[str, float]]:
        """
        Assess overall signal quality and confidence
        
        Args:
            rr_ms: Cleaned RR intervals
            artifact_pct: Percentage of artifacts detected
            
        Returns:
            Quality assessment dictionary
        """
        # Quality factors
        duration_score = min(len(rr_ms) / 50, 1.0)  # Prefer 50+ beats
        artifact_score = max(0, 1 - artifact_pct / 30)  # Penalize >30% artifacts
        
        # Overall confidence
        confidence = (duration_score * artifact_score) * 100
        
        # Quality categories
        if confidence >= 80 and artifact_pct < 10:
            quality = "excellent"
            color = "#4CAF50"
        elif confidence >= 60 and artifact_pct < 20:
            quality = "good"
            color = "#8BC34A"
        elif confidence >= 40 and artifact_pct < 30:
            quality = "fair"
            color = "#FFC107"
        else:
            quality = "poor"
            color = "#F44336"
        
        return {
            'quality': quality,
            'confidence': confidence,
            'artifact_pct': artifact_pct,
            'color': color,
            'beats_count': len(rr_ms),
            'duration_score': duration_score * 100,
            'artifact_score': artifact_score * 100
        }
    
    def generate_clinical_flags(self, metrics: Dict, cycle_info: Dict, 
                               user_baselines: Dict = None) -> List[Dict]:
        """
        Generate clinical flags based on rule engine
        
        Args:
            metrics: Combined HRV metrics
            cycle_info: Cycle phase information
            user_baselines: User's historical baselines
            
        Returns:
            List of clinical flags
        """
        flags = []
        
        hr = metrics.get('mean_hr_bpm', 0)
        rmssd = metrics.get('rmssd_ms', 0)
        artifact_pct = metrics.get('artifact_pct', 0)
        
        # Flag 1: Poor signal quality
        if artifact_pct > 30:
            flags.append({
                'type': 'signal_quality',
                'severity': 'low',
                'message': 'Poor signal quality detected. Try better finger placement and lighting.',
                'confidence': 90,
                'actionable': True
            })
        
        # Flag 2: Elevated resting heart rate
        if hr > 100:
            flags.append({
                'type': 'elevated_hr',
                'severity': 'medium',
                'message': f'Elevated heart rate ({hr:.0f} BPM). Consider rest and hydration.',
                'confidence': 85,
                'actionable': True
            })
        elif hr > 90:
            flags.append({
                'type': 'elevated_hr',
                'severity': 'low',
                'message': f'Slightly elevated heart rate ({hr:.0f} BPM). Monitor trends.',
                'confidence': 75,
                'actionable': False
            })
        
        # Flag 3: Low HRV
        if rmssd < 15:
            flags.append({
                'type': 'low_hrv',
                'severity': 'medium',
                'message': 'Low heart rate variability may indicate stress or fatigue.',
                'confidence': 80,
                'actionable': True
            })
        
        # Flag 4: Cycle-aware flags
        phase = cycle_info.get('phase', 'unknown')
        if phase == 'luteal' and hr < 60:
            flags.append({
                'type': 'cycle_anomaly',
                'severity': 'low',
                'message': 'Lower than expected heart rate for luteal phase.',
                'confidence': 70,
                'actionable': False
            })
        
        return flags
    
    def process_ppg_session(self, rr_intervals: List[float], 
                           last_period_start: str = None,
                           cycle_length: int = 28,
                           user_baselines: Dict = None) -> Dict:
        """
        Complete PPG session processing with all metrics and analysis
        
        Args:
            rr_intervals: Raw RR intervals from PPG
            last_period_start: Date of last period
            cycle_length: Menstrual cycle length
            user_baselines: User's baseline metrics
            
        Returns:
            Complete analysis results
        """
        # Clean RR intervals
        clean_rr, artifact_pct = self.clean_rr_intervals(rr_intervals)
        
        # Assess signal quality
        quality_assessment = self.assess_signal_quality(clean_rr, artifact_pct)
        
        # Compute metrics if sufficient data
        results = {
            'timestamp': datetime.now().isoformat(),
            'quality': quality_assessment,
            'raw_beats': len(rr_intervals),
            'clean_beats': len(clean_rr)
        }
        
        if len(clean_rr) >= 5:
            # Time-domain metrics
            time_metrics = self.compute_time_domain_metrics(clean_rr)
            results.update(time_metrics)
            
            # Frequency-domain metrics (if sufficient data)
            if len(clean_rr) >= 30:
                freq_metrics = self.compute_frequency_domain_metrics(clean_rr)
                results.update(freq_metrics)
            else:
                freq_metrics = {}
            
            # Cycle information
            if last_period_start:
                cycle_info = self.estimate_cycle_phase(last_period_start, cycle_length)
                results['cycle_info'] = cycle_info
            else:
                cycle_info = {}
            
            # Derived indices
            derived_indices = self.compute_derived_indices(time_metrics, freq_metrics, user_baselines)
            results['indices'] = derived_indices
            
            # Clinical flags
            combined_metrics = {**time_metrics, **freq_metrics, 'artifact_pct': artifact_pct}
            flags = self.generate_clinical_flags(combined_metrics, cycle_info, user_baselines)
            results['flags'] = flags
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = AdvancedPPGProcessor()
    
    # Simulate RR intervals (in milliseconds)
    # Normal sinus rhythm around 70 BPM with some variability
    base_rr = 857  # 70 BPM
    rr_intervals = []
    for i in range(100):
        # Add physiological variability
        variability = np.random.normal(0, 50)  # RMSSD ~35ms
        rr = base_rr + variability
        rr_intervals.append(rr)
    
    # Add some artifacts
    rr_intervals[20] = 400  # Artifact
    rr_intervals[50] = 1500  # Artifact
    
    # Process session
    results = processor.process_ppg_session(
        rr_intervals=rr_intervals,
        last_period_start="2025-10-01",
        cycle_length=28
    )
    
    print("=== PulseHER Advanced PPG Analysis ===")
    print(f"Signal Quality: {results['quality']['quality']} ({results['quality']['confidence']:.1f}%)")
    print(f"Heart Rate: {results.get('mean_hr_bpm', 'N/A'):.1f} BPM")
    print(f"RMSSD: {results.get('rmssd_ms', 'N/A'):.1f} ms")
    print(f"SDNN: {results.get('sdnn_ms', 'N/A'):.1f} ms")
    print(f"Cycle Phase: {results.get('cycle_info', {}).get('phase', 'Unknown')}")
    print(f"ABI: {results.get('indices', {}).get('abi', 'N/A'):.1f}")
    print(f"CVR: {results.get('indices', {}).get('cvr', 'N/A'):.1f}")
    print(f"Flags: {len(results.get('flags', []))}")
    
    for flag in results.get('flags', []):
        print(f"  - {flag['severity'].upper()}: {flag['message']}")