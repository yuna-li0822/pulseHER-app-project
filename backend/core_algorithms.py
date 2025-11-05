"""
PulseHER Core Algorithms - Clinical-Grade Processing
Implements the exact algorithms from the comprehensive blueprint
"""

import math
import statistics
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PulseHERCore:
    """
    Core processing engine implementing all blueprint algorithms
    """
    
    def __init__(self):
        # Algorithm parameters
        self.min_rr_ms = 250  # 240 BPM max
        self.max_rr_ms = 2000  # 30 BPM min
        self.artifact_threshold = 0.2  # 20% change threshold
        
        # Frequency domain parameters
        self.vlf_range = (0.003, 0.04)  # Very Low Frequency
        self.lf_range = (0.04, 0.15)   # Low Frequency
        self.hf_range = (0.15, 0.4)    # High Frequency
        
    def detect_beats_from_ppg(self, ppg_signal, sampling_rate=30):
        """
        Beat detection from raw PPG signal
        Blueprint Step 5.1: Peak detection → RR intervals
        """
        try:
            # Bandpass filter (0.5-8 Hz for PPG)
            nyquist = sampling_rate / 2
            low = 0.5 / nyquist
            high = 8.0 / nyquist
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_ppg = signal.filtfilt(b, a, ppg_signal)
            
            # Find peaks with adaptive threshold
            peaks, properties = signal.find_peaks(
                filtered_ppg,
                height=np.std(filtered_ppg) * 0.3,
                distance=int(sampling_rate * 0.4)  # Min 0.4s between beats
            )
            
            # Convert peak indices to RR intervals (ms)
            rr_intervals = []
            for i in range(1, len(peaks)):
                interval_samples = peaks[i] - peaks[i-1]
                interval_ms = (interval_samples / sampling_rate) * 1000
                rr_intervals.append(interval_ms)
            
            logger.info(f"Detected {len(rr_intervals)} RR intervals from PPG")
            return rr_intervals
            
        except Exception as e:
            logger.error(f"Beat detection error: {e}")
            return []
    
    def clean_rr_intervals(self, rr_ms):
        """
        Blueprint Step 5.2: RR cleaning algorithm
        Removes physiologically impossible intervals and artifacts
        """
        if not rr_ms or len(rr_ms) < 3:
            return [], 100.0  # 100% artifact rate
        
        original_count = len(rr_ms)
        
        # Step 1: Remove impossible intervals
        valid_rr = [r for r in rr_ms if self.min_rr_ms <= r <= self.max_rr_ms]
        
        # Step 2: Remove large jumps (artifacts)
        cleaned = []
        artifacts = 0
        
        for i, r in enumerate(valid_rr):
            if i == 0:
                cleaned.append(r)
                continue
                
            prev = cleaned[-1] if cleaned else valid_rr[0]
            change_pct = abs(r - prev) / prev
            
            if change_pct > self.artifact_threshold:
                artifacts += 1
                # Skip artifact or interpolate
                continue
            
            cleaned.append(r)
        
        # Calculate artifact percentage
        removed_count = original_count - len(cleaned)
        artifact_pct = (removed_count / original_count) * 100 if original_count > 0 else 100
        
        logger.info(f"RR cleaning: {len(cleaned)}/{original_count} intervals kept, {artifact_pct:.1f}% artifacts")
        return cleaned, artifact_pct
    
    def compute_time_domain_metrics(self, rr_ms):
        """
        Blueprint Step 5.3: Time domain HRV metrics
        Returns: RMSSD, SDNN, pNN50, mean HR
        """
        if len(rr_ms) < 3:
            return None
        
        try:
            # Successive differences
            diffs = [rr_ms[i+1] - rr_ms[i] for i in range(len(rr_ms)-1)]
            
            # RMSSD (Root Mean Square of Successive Differences)
            rmssd = math.sqrt(sum(d*d for d in diffs) / len(diffs))
            
            # SDNN (Standard Deviation of NN intervals)
            sdnn = statistics.pstdev(rr_ms)
            
            # pNN50 (percentage of successive RR intervals >50ms apart)
            pnn50 = sum(1 for d in diffs if abs(d) > 50) / len(diffs) * 100
            
            # Mean heart rate
            mean_rr = sum(rr_ms) / len(rr_ms)
            mean_hr = 60000 / mean_rr  # Convert ms to BPM
            
            return {
                'rmssd_ms': round(rmssd, 2),
                'sdnn_ms': round(sdnn, 2),
                'pnn50_pct': round(pnn50, 2),
                'mean_hr_bpm': round(mean_hr, 1),
                'mean_rr_ms': round(mean_rr, 2),
                'valid_intervals': len(rr_ms)
            }
            
        except Exception as e:
            logger.error(f"Time domain computation error: {e}")
            return None
    
    def compute_frequency_domain_metrics(self, rr_ms, sampling_rate=4):
        """
        Blueprint Step 5.4: Frequency domain analysis
        Returns: VLF, LF, HF power, LF/HF ratio
        """
        if len(rr_ms) < 10:
            return None
        
        try:
            # Resample RR intervals to regular time series
            timestamps = np.cumsum([0] + rr_ms[:-1]) / 1000.0  # Convert to seconds
            interpolated_time = np.arange(0, timestamps[-1], 1/sampling_rate)
            
            # Interpolate RR series
            rr_interpolated = np.interp(interpolated_time, timestamps, rr_ms)
            
            # Detrend and apply window
            rr_detrended = signal.detrend(rr_interpolated)
            window = signal.hann(len(rr_detrended))
            rr_windowed = rr_detrended * window
            
            # FFT and power spectral density
            freqs = fftfreq(len(rr_windowed), 1/sampling_rate)
            fft_vals = np.abs(fft(rr_windowed))**2
            
            # Calculate power in frequency bands
            vlf_power = self._calculate_band_power(freqs, fft_vals, self.vlf_range)
            lf_power = self._calculate_band_power(freqs, fft_vals, self.lf_range)
            hf_power = self._calculate_band_power(freqs, fft_vals, self.hf_range)
            
            total_power = vlf_power + lf_power + hf_power
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
            
            return {
                'vlf_power': round(vlf_power, 2),
                'lf_power': round(lf_power, 2),
                'hf_power': round(hf_power, 2),
                'total_power': round(total_power, 2),
                'lf_hf_ratio': round(lf_hf_ratio, 3),
                'lf_norm': round((lf_power / (lf_power + hf_power)) * 100, 1) if (lf_power + hf_power) > 0 else 0,
                'hf_norm': round((hf_power / (lf_power + hf_power)) * 100, 1) if (lf_power + hf_power) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Frequency domain computation error: {e}")
            return None
    
    def _calculate_band_power(self, freqs, psd, freq_range):
        """Helper: Calculate power in specific frequency band"""
        band_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
        return np.sum(psd[band_indices]) if len(band_indices) > 0 else 0.0
    
    def compute_cycle_aware_baseline(self, user_history, current_phase):
        """
        Blueprint Step 5.4: Per-phase baseline computation
        Returns baseline statistics for current cycle phase
        """
        if not user_history or current_phase not in user_history:
            return None
        
        phase_data = user_history[current_phase]
        
        if len(phase_data) < 3:  # Need minimum data points
            return None
        
        baseline = {}
        for metric in ['mean_hr_bpm', 'rmssd_ms', 'sdnn_ms', 'lf_hf_ratio']:
            values = [d.get(metric) for d in phase_data if d.get(metric) is not None]
            if len(values) >= 3:
                baseline[metric] = {
                    'mean': statistics.mean(values),
                    'std': statistics.pstdev(values),
                    'count': len(values)
                }
        
        return baseline
    
    def compute_clinical_indices(self, metrics, baseline=None, user_profile=None):
        """
        Blueprint Step 5.5: Compute ABI, CVR, CSI indices
        """
        if not metrics:
            return None
        
        try:
            # Get baseline z-scores if available
            z_scores = self._compute_z_scores(metrics, baseline) if baseline else {}
            
            # ABI (Autonomic Balance Index) - Blueprint formula
            z_rmssd = z_scores.get('rmssd_ms', 0)
            z_hr = z_scores.get('mean_hr_bpm', 0)
            
            # Lower ABI = more sympathetic dominance
            abi_raw = 0.6 * (-z_rmssd) + 0.4 * z_hr
            abi = max(0, min(100, 50 + (abi_raw * 10)))  # Scale to 0-100
            
            # CVR (Cardiovascular Risk) - Multi-factor assessment
            hr_risk = 1 if metrics['mean_hr_bpm'] > 90 else 0
            hrv_risk = 1 if metrics['rmssd_ms'] < 20 else 0
            lf_hf_risk = 1 if metrics.get('lf_hf_ratio', 0) > 4 else 0
            
            cvr_raw = (hr_risk + hrv_risk + lf_hf_risk) / 3
            cvr = round(cvr_raw * 100, 1)
            
            # CSI (Cardiac Stress Index) - Sympathetic activation
            stress_indicators = []
            stress_indicators.append(metrics['mean_hr_bpm'] / 100)  # Normalized HR
            stress_indicators.append(max(0, (100 - metrics['rmssd_ms']) / 100))  # Inverse RMSSD
            stress_indicators.append(min(1, metrics.get('lf_hf_ratio', 0) / 5))  # LF/HF ratio
            
            csi = round(statistics.mean(stress_indicators) * 100, 1)
            
            return {
                'abi': round(abi, 1),
                'cvr': cvr,
                'csi': csi,
                'z_scores': z_scores,
                'explanation': {
                    'abi': f"Autonomic balance: {'Sympathetic' if abi > 60 else 'Balanced' if abi > 40 else 'Parasympathetic'} dominance",
                    'cvr': f"Cardiovascular risk: {'High' if cvr > 66 else 'Moderate' if cvr > 33 else 'Low'}",
                    'csi': f"Cardiac stress: {'High' if csi > 70 else 'Moderate' if csi > 40 else 'Low'}"
                }
            }
            
        except Exception as e:
            logger.error(f"Clinical indices computation error: {e}")
            return None
    
    def _compute_z_scores(self, metrics, baseline):
        """Helper: Compute z-scores relative to personal baseline"""
        z_scores = {}
        
        for metric in ['mean_hr_bpm', 'rmssd_ms', 'sdnn_ms', 'lf_hf_ratio']:
            if metric in metrics and metric in baseline:
                value = metrics[metric]
                mean = baseline[metric]['mean']
                std = baseline[metric]['std']
                
                if std > 0:
                    z_scores[metric] = (value - mean) / std
                else:
                    z_scores[metric] = 0
        
        return z_scores
    
    def generate_clinical_flags(self, metrics, indices, artifact_pct):
        """
        Blueprint Step 5.6: Rule engine for clinical flags
        """
        flags = []
        
        # Quality flags
        if artifact_pct > 20:
            flags.append({
                'type': 'quality',
                'severity': 'warning',
                'message': f'High artifact rate ({artifact_pct:.1f}%). Consider re-recording.',
                'confidence': 0.9
            })
        
        # Heart rate flags
        if metrics['mean_hr_bpm'] > 100:
            flags.append({
                'type': 'tachycardia',
                'severity': 'alert',
                'message': f'Elevated heart rate ({metrics["mean_hr_bpm"]:.1f} BPM). Consider medical evaluation.',
                'confidence': 0.8
            })
        
        if metrics['mean_hr_bpm'] < 50:
            flags.append({
                'type': 'bradycardia', 
                'severity': 'alert',
                'message': f'Low heart rate ({metrics["mean_hr_bpm"]:.1f} BPM). Consider medical evaluation.',
                'confidence': 0.8
            })
        
        # HRV flags
        if metrics['rmssd_ms'] < 15:
            flags.append({
                'type': 'low_hrv',
                'severity': 'info',
                'message': 'Low heart rate variability detected. May indicate stress or fatigue.',
                'confidence': 0.7
            })
        
        # Clinical index flags
        if indices:
            if indices['cvr'] > 66:
                flags.append({
                    'type': 'cardiovascular_risk',
                    'severity': 'alert',
                    'message': f'Elevated cardiovascular risk indicators (CVR: {indices["cvr"]}%). Recommend clinical evaluation.',
                    'confidence': 0.75
                })
            
            if indices['csi'] > 80:
                flags.append({
                    'type': 'high_stress',
                    'severity': 'warning',
                    'message': f'High cardiac stress index ({indices["csi"]}%). Consider stress management.',
                    'confidence': 0.7
                })
        
        return flags
    
    def estimate_cycle_phase(self, last_period_date, cycle_length=28):
        """
        Blueprint: Estimate current menstrual cycle phase
        """
        if not last_period_date:
            return 'unknown'
        
        try:
            if isinstance(last_period_date, str):
                last_period = datetime.fromisoformat(last_period_date.replace('Z', '+00:00'))
            else:
                last_period = last_period_date
            
            days_since = (datetime.now() - last_period).days
            cycle_day = days_since % cycle_length
            
            # Standard phase mapping
            if 1 <= cycle_day <= 5:
                return 'menstrual'
            elif 6 <= cycle_day <= cycle_length // 2 - 2:
                return 'follicular'
            elif cycle_length // 2 - 1 <= cycle_day <= cycle_length // 2 + 1:
                return 'ovulation'
            else:
                return 'luteal'
                
        except Exception as e:
            logger.error(f"Cycle phase estimation error: {e}")
            return 'unknown'
    
    def process_session_complete(self, rr_intervals, user_profile=None, baseline=None):
        """
        Complete processing pipeline - Blueprint integration
        """
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'processing_version': '4.0'
        }
        
        # Step 1: Clean RR intervals
        cleaned_rr, artifact_pct = self.clean_rr_intervals(rr_intervals)
        result['rr_quality'] = {
            'original_count': len(rr_intervals),
            'cleaned_count': len(cleaned_rr),
            'artifact_pct': artifact_pct
        }
        
        if len(cleaned_rr) < 3:
            result['error'] = 'Insufficient valid RR intervals for analysis'
            return result
        
        # Step 2: Time domain metrics
        time_metrics = self.compute_time_domain_metrics(cleaned_rr)
        if not time_metrics:
            result['error'] = 'Time domain analysis failed'
            return result
        
        result['time_domain'] = time_metrics
        
        # Step 3: Frequency domain metrics
        freq_metrics = self.compute_frequency_domain_metrics(cleaned_rr)
        if freq_metrics:
            result['frequency_domain'] = freq_metrics
            # Merge for clinical indices
            combined_metrics = {**time_metrics, **freq_metrics}
        else:
            combined_metrics = time_metrics
        
        # Step 4: Clinical indices
        indices = self.compute_clinical_indices(combined_metrics, baseline, user_profile)
        if indices:
            result['clinical_indices'] = indices
        
        # Step 5: Clinical flags
        flags = self.generate_clinical_flags(combined_metrics, indices, artifact_pct)
        result['clinical_flags'] = flags
        
        # Step 6: Cycle awareness
        if user_profile and 'last_period_date' in user_profile:
            cycle_length = user_profile.get('cycle_length', 28)
            phase = self.estimate_cycle_phase(user_profile['last_period_date'], cycle_length)
            result['cycle_info'] = {
                'phase': phase,
                'cycle_length': cycle_length
            }
        
        return result


# Singleton instance for use across the application
pulse_core = PulseHERCore()