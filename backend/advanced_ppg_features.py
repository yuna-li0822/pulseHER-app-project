"""
Advanced PPG Feature Extraction Module
Implements sophisticated signal analysis for raw PPG waveforms
Extracts pulse transit time, morphology features, and advanced HRV metrics
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class AdvancedPPGFeatureExtractor:
    """
    Advanced feature extraction from raw PPG signals
    Implements cutting-edge cardiovascular signal analysis
    """
    
    def __init__(self, sampling_rate: float = 30.0):
        """
        Initialize advanced PPG feature extractor
        
        Args:
            sampling_rate: PPG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        
    def extract_comprehensive_features(self, ppg_signal: np.ndarray, 
                                     user_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract comprehensive feature set from raw PPG signal
        
        Args:
            ppg_signal: Raw PPG waveform data
            user_metadata: Optional user information (age, sex, cycle_phase, etc.)
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Validate input
        if len(ppg_signal) < 300:  # Need at least 10 seconds at 30 Hz
            return self._get_empty_features()
        
        # Preprocess signal
        clean_signal = self._preprocess_signal(ppg_signal)
        
        # Extract different feature categories
        features.update(self._extract_pulse_morphology_features(clean_signal))
        features.update(self._extract_pulse_transit_time_features(clean_signal))
        features.update(self._extract_advanced_hrv_features(clean_signal))
        features.update(self._extract_frequency_domain_features(clean_signal))
        features.update(self._extract_nonlinear_features(clean_signal))
        features.update(self._extract_physiological_indices(clean_signal))
        
        # Add metadata-enhanced features
        if user_metadata:
            features.update(self._extract_personalized_features(features, user_metadata))
        
        return features
    
    def _preprocess_signal(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Advanced signal preprocessing with noise removal and normalization"""
        try:
            # Remove DC component
            signal_ac = ppg_signal - np.mean(ppg_signal)
            
            # Bandpass filter (0.5-8 Hz for cardiovascular signals)
            sos = signal.butter(4, [0.5, 8.0], btype='band', 
                              fs=self.sampling_rate, output='sos')
            filtered_signal = signal.sosfilt(sos, signal_ac)
            
            # Remove artifacts using median filter
            denoised = signal.medfilt(filtered_signal, kernel_size=5)
            
            # Normalize signal
            normalized = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised))
            
            return normalized
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return ppg_signal
    
    def _extract_pulse_morphology_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract detailed pulse shape and morphology features"""
        features = {}
        
        try:
            # Detect peaks (systolic peaks)
            peaks, peak_properties = signal.find_peaks(
                ppg_signal, 
                height=0.3, 
                distance=int(0.4 * self.sampling_rate)  # Min 0.4s between peaks
            )
            
            if len(peaks) < 3:
                return self._get_empty_morphology_features()
            
            # Calculate pulse characteristics for each pulse
            pulse_widths = []
            systolic_amplitudes = []
            diastolic_amplitudes = []
            pulse_asymmetries = []
            
            for i in range(len(peaks) - 1):
                start_idx = peaks[i]
                end_idx = peaks[i + 1]
                pulse_segment = ppg_signal[start_idx:end_idx]
                
                # Pulse width (time between consecutive peaks)
                pulse_width = (end_idx - start_idx) / self.sampling_rate
                pulse_widths.append(pulse_width)
                
                # Systolic amplitude (peak height)
                systolic_amp = ppg_signal[start_idx]
                systolic_amplitudes.append(systolic_amp)
                
                # Diastolic amplitude (minimum between peaks)
                if len(pulse_segment) > 0:
                    diastolic_amp = np.min(pulse_segment)
                    diastolic_amplitudes.append(diastolic_amp)
                    
                    # Pulse asymmetry (skewness of pulse shape)
                    pulse_asymmetry = self._calculate_pulse_asymmetry(pulse_segment)
                    pulse_asymmetries.append(pulse_asymmetry)
            
            # Statistical features of morphology
            features['mean_pulse_width'] = np.mean(pulse_widths)
            features['std_pulse_width'] = np.std(pulse_widths)
            features['mean_systolic_amplitude'] = np.mean(systolic_amplitudes)
            features['std_systolic_amplitude'] = np.std(systolic_amplitudes)
            features['mean_diastolic_amplitude'] = np.mean(diastolic_amplitudes)
            features['pulse_pressure_ratio'] = np.mean(systolic_amplitudes) / (np.mean(diastolic_amplitudes) + 0.001)
            features['mean_pulse_asymmetry'] = np.mean(pulse_asymmetries)
            
            # Advanced morphology indices
            features['pulse_regularity_index'] = 1.0 / (1.0 + np.std(pulse_widths))
            features['amplitude_variation_index'] = np.std(systolic_amplitudes) / np.mean(systolic_amplitudes)
            
            return features
            
        except Exception as e:
            print(f"Morphology extraction failed: {e}")
            return self._get_empty_morphology_features()
    
    def _extract_pulse_transit_time_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract pulse transit time and related vascular features"""
        features = {}
        
        try:
            # Detect systolic peaks
            peaks, _ = signal.find_peaks(ppg_signal, height=0.3, distance=int(0.4 * self.sampling_rate))
            
            if len(peaks) < 3:
                return {'pulse_transit_time': None, 'arterial_stiffness_index': None}
            
            # Calculate pulse transit times (time between consecutive peaks)
            transit_times = np.diff(peaks) / self.sampling_rate
            
            features['mean_pulse_transit_time'] = np.mean(transit_times)
            features['std_pulse_transit_time'] = np.std(transit_times)
            features['pulse_transit_time_variability'] = np.std(transit_times) / np.mean(transit_times)
            
            # Arterial stiffness index (inverse relationship with PTT)
            mean_ptt = np.mean(transit_times)
            features['arterial_stiffness_index'] = 1.0 / mean_ptt if mean_ptt > 0 else None
            
            # Pulse wave velocity approximation (requires calibration)
            # PWV ≈ distance / PTT (distance estimated from demographics)
            features['estimated_pwv'] = 1.5 / mean_ptt if mean_ptt > 0 else None  # Rough approximation
            
            return features
            
        except Exception as e:
            print(f"PTT extraction failed: {e}")
            return {'pulse_transit_time': None, 'arterial_stiffness_index': None}
    
    def _extract_advanced_hrv_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract advanced HRV features beyond basic RMSSD/SDNN"""
        features = {}
        
        try:
            # Get RR intervals from PPG peaks
            rr_intervals = self._extract_rr_intervals(ppg_signal)
            
            if len(rr_intervals) < 10:
                return self._get_empty_hrv_features()
            
            # Time-domain HRV measures
            features.update(self._calculate_time_domain_hrv(rr_intervals))
            
            # Geometric HRV measures
            features.update(self._calculate_geometric_hrv(rr_intervals))
            
            # Poincaré plot features
            features.update(self._calculate_poincare_features(rr_intervals))
            
            return features
            
        except Exception as e:
            print(f"Advanced HRV extraction failed: {e}")
            return self._get_empty_hrv_features()
    
    def _extract_frequency_domain_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive frequency domain features"""
        features = {}
        
        try:
            # Get RR intervals for HRV frequency analysis
            rr_intervals = self._extract_rr_intervals(ppg_signal)
            
            if len(rr_intervals) < 10:
                return self._get_empty_frequency_features()
            
            # Interpolate RR intervals to uniform sampling
            rr_times = np.cumsum(rr_intervals)
            interp_freq = 4.0  # 4 Hz interpolation
            uniform_time = np.arange(0, rr_times[-1], 1.0/interp_freq)
            uniform_rr = np.interp(uniform_time, rr_times, rr_intervals)
            
            # Power spectral density
            f, psd = signal.welch(uniform_rr, fs=interp_freq, nperseg=min(256, len(uniform_rr)//4))
            
            # Define frequency bands
            vlf_mask = (f >= 0.0033) & (f < 0.04)
            lf_mask = (f >= 0.04) & (f < 0.15)
            hf_mask = (f >= 0.15) & (f < 0.4)
            
            # Calculate power in each band
            vlf_power = np.trapz(psd[vlf_mask], f[vlf_mask])
            lf_power = np.trapz(psd[lf_mask], f[lf_mask])
            hf_power = np.trapz(psd[hf_mask], f[hf_mask])
            total_power = vlf_power + lf_power + hf_power
            
            # Frequency domain features
            features['vlf_power'] = vlf_power
            features['lf_power'] = lf_power
            features['hf_power'] = hf_power
            features['total_power'] = total_power
            features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else None
            features['normalized_lf'] = lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else None
            features['normalized_hf'] = hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else None
            
            # Peak frequencies
            lf_peak_freq = f[lf_mask][np.argmax(psd[lf_mask])] if len(f[lf_mask]) > 0 else None
            hf_peak_freq = f[hf_mask][np.argmax(psd[hf_mask])] if len(f[hf_mask]) > 0 else None
            
            features['lf_peak_frequency'] = lf_peak_freq
            features['hf_peak_frequency'] = hf_peak_freq
            
            return features
            
        except Exception as e:
            print(f"Frequency domain extraction failed: {e}")
            return self._get_empty_frequency_features()
    
    def _extract_nonlinear_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract nonlinear dynamics features"""
        features = {}
        
        try:
            rr_intervals = self._extract_rr_intervals(ppg_signal)
            
            if len(rr_intervals) < 20:
                return self._get_empty_nonlinear_features()
            
            # Approximate Entropy
            features['approximate_entropy'] = self._calculate_approximate_entropy(rr_intervals)
            
            # Sample Entropy
            features['sample_entropy'] = self._calculate_sample_entropy(rr_intervals)
            
            # Detrended Fluctuation Analysis
            features['dfa_alpha1'], features['dfa_alpha2'] = self._calculate_dfa(rr_intervals)
            
            # Correlation Dimension
            features['correlation_dimension'] = self._calculate_correlation_dimension(rr_intervals)
            
            return features
            
        except Exception as e:
            print(f"Nonlinear feature extraction failed: {e}")
            return self._get_empty_nonlinear_features()
    
    def _extract_physiological_indices(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract novel physiological indices"""
        features = {}
        
        try:
            # Autonomic Balance Index (enhanced version)
            rr_intervals = self._extract_rr_intervals(ppg_signal)
            if len(rr_intervals) > 10:
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
                sdnn = np.std(rr_intervals)
                features['enhanced_abi'] = sdnn / rmssd if rmssd > 0 else None
                
                # Baroreflex Sensitivity approximation
                features['baroreflex_sensitivity'] = self._estimate_baroreflex_sensitivity(rr_intervals)
                
                # Endothelial Function Index
                features['endothelial_function_index'] = self._estimate_endothelial_function(ppg_signal)
            
            return features
            
        except Exception as e:
            print(f"Physiological indices extraction failed: {e}")
            return {}
    
    def _extract_personalized_features(self, base_features: Dict, 
                                     metadata: Dict) -> Dict[str, float]:
        """Extract personalized features based on user metadata"""
        features = {}
        
        try:
            age = metadata.get('age', 30)
            sex = metadata.get('sex', 'female')
            cycle_phase = metadata.get('cycle_phase', 'unknown')
            
            # Age-adjusted features
            if 'mean_pulse_transit_time' in base_features and base_features['mean_pulse_transit_time']:
                expected_ptt = 0.8 - (age - 20) * 0.002  # Age adjustment
                features['age_adjusted_ptt'] = base_features['mean_pulse_transit_time'] / expected_ptt
            
            # Sex-specific adjustments
            if sex == 'female' and 'lf_hf_ratio' in base_features:
                # Females typically have higher HF power
                features['sex_adjusted_lf_hf'] = base_features['lf_hf_ratio'] * 1.1
            
            # Menstrual cycle adjustments
            if cycle_phase in ['menstrual', 'follicular', 'ovulatory', 'luteal']:
                cycle_multipliers = {
                    'menstrual': 0.9,
                    'follicular': 1.0,
                    'ovulatory': 1.1,
                    'luteal': 0.95
                }
                
                if 'enhanced_abi' in base_features and base_features['enhanced_abi']:
                    features['cycle_adjusted_abi'] = (base_features['enhanced_abi'] * 
                                                    cycle_multipliers[cycle_phase])
            
            return features
            
        except Exception as e:
            print(f"Personalized feature extraction failed: {e}")
            return {}
    
    # Helper methods for detailed calculations
    def _extract_rr_intervals(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Extract RR intervals from PPG signal"""
        peaks, _ = signal.find_peaks(ppg_signal, height=0.3, distance=int(0.4 * self.sampling_rate))
        if len(peaks) < 2:
            return np.array([])
        
        rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # Convert to milliseconds
        return rr_intervals[rr_intervals > 0]  # Remove invalid intervals
    
    def _calculate_pulse_asymmetry(self, pulse_segment: np.ndarray) -> float:
        """Calculate pulse shape asymmetry"""
        if len(pulse_segment) < 3:
            return 0.0
        
        peak_idx = np.argmax(pulse_segment)
        if peak_idx == 0 or peak_idx == len(pulse_segment) - 1:
            return 0.0
        
        # Asymmetry as ratio of left/right areas under curve
        left_area = np.trapz(pulse_segment[:peak_idx])
        right_area = np.trapz(pulse_segment[peak_idx:])
        
        return left_area / (right_area + 0.001)
    
    def _calculate_time_domain_hrv(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive time-domain HRV features"""
        if len(rr_intervals) < 2:
            return {}
        
        features = {}
        
        # Basic measures
        features['mean_rr'] = np.mean(rr_intervals)
        features['sdnn'] = np.std(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        
        # Percentage measures
        diff_rr = np.abs(np.diff(rr_intervals))
        features['pnn50'] = np.sum(diff_rr > 50) / len(diff_rr) * 100
        features['pnn20'] = np.sum(diff_rr > 20) / len(diff_rr) * 100
        
        # Advanced measures
        features['cv_rr'] = features['sdnn'] / features['mean_rr']  # Coefficient of variation
        features['rr_tri_index'] = len(rr_intervals) / np.max(np.histogram(rr_intervals, bins=32)[0])
        
        return features
    
    def _calculate_geometric_hrv(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate geometric HRV measures"""
        if len(rr_intervals) < 20:
            return {}
        
        features = {}
        
        # Triangular index
        hist, _ = np.histogram(rr_intervals, bins=32)
        features['triangular_index'] = len(rr_intervals) / np.max(hist)
        
        # TINN (Triangular Interpolation of NN interval histogram)
        features['tinn'] = np.max(rr_intervals) - np.min(rr_intervals)
        
        return features
    
    def _calculate_poincare_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate Poincaré plot features"""
        if len(rr_intervals) < 3:
            return {}
        
        # Poincaré plot: RR[n] vs RR[n+1]
        rr1 = rr_intervals[:-1]
        rr2 = rr_intervals[1:]
        
        # SD1 and SD2
        diff_rr = rr2 - rr1
        sum_rr = rr2 + rr1
        
        sd1 = np.std(diff_rr) / np.sqrt(2)
        sd2 = np.std(sum_rr) / np.sqrt(2)
        
        features = {
            'poincare_sd1': sd1,
            'poincare_sd2': sd2,
            'poincare_ratio': sd2 / sd1 if sd1 > 0 else None
        }
        
        return features
    
    def _calculate_approximate_entropy(self, rr_intervals: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate Approximate Entropy"""
        if len(rr_intervals) < 10:
            return None
        
        if r is None:
            r = 0.2 * np.std(rr_intervals)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([rr_intervals[i:i + m] for i in range(len(rr_intervals) - m + 1)])
            C = np.zeros(len(patterns))
            
            for i, pattern in enumerate(patterns):
                matches = sum(1 for other_pattern in patterns 
                             if _maxdist(pattern, other_pattern, m) <= r)
                C[i] = matches / len(patterns)
            
            return np.mean(np.log(C))
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return None
    
    def _calculate_sample_entropy(self, rr_intervals: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate Sample Entropy"""
        if len(rr_intervals) < 10:
            return None
        
        if r is None:
            r = 0.2 * np.std(rr_intervals)
        
        try:
            # Implementation of SampEn algorithm
            N = len(rr_intervals)
            
            def _template_match(template, data, r, m):
                matches = 0
                for i in range(len(data) - m + 1):
                    if max(abs(template[j] - data[i + j]) for j in range(m)) <= r:
                        matches += 1
                return matches
            
            A = 0
            B = 0
            
            for i in range(N - m):
                template_m = rr_intervals[i:i + m]
                template_m1 = rr_intervals[i:i + m + 1]
                
                remaining_data = rr_intervals[i + 1:]
                
                B += _template_match(template_m, remaining_data, r, m)
                A += _template_match(template_m1, remaining_data, r, m + 1)
            
            return -np.log(A / B) if B > 0 else None
        except:
            return None
    
    def _calculate_dfa(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """Calculate Detrended Fluctuation Analysis alpha1 and alpha2"""
        if len(rr_intervals) < 16:
            return None, None
        
        try:
            # Integrate the signal
            y = np.cumsum(rr_intervals - np.mean(rr_intervals))
            
            # Define box sizes
            scales = np.unique(np.logspace(0.5, np.log10(len(rr_intervals) // 4), 20).astype(int))
            fluctuations = []
            
            for scale in scales:
                # Divide into boxes
                n_boxes = len(y) // scale
                if n_boxes < 4:
                    continue
                    
                box_fluctuations = []
                for i in range(n_boxes):
                    box_data = y[i * scale:(i + 1) * scale]
                    # Linear detrending
                    x = np.arange(len(box_data))
                    coeffs = np.polyfit(x, box_data, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = box_data - trend
                    box_fluctuations.append(np.sqrt(np.mean(detrended ** 2)))
                
                fluctuations.append(np.mean(box_fluctuations))
            
            if len(fluctuations) < 8:
                return None, None
            
            # Fit power law
            log_scales = np.log10(scales[:len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)
            
            # Split into short-term (alpha1) and long-term (alpha2)
            mid_point = len(log_scales) // 2
            
            # Alpha1 (short-term scaling)
            alpha1 = np.polyfit(log_scales[:mid_point], log_fluctuations[:mid_point], 1)[0]
            
            # Alpha2 (long-term scaling)
            alpha2 = np.polyfit(log_scales[mid_point:], log_fluctuations[mid_point:], 1)[0]
            
            return alpha1, alpha2
            
        except:
            return None, None
    
    def _calculate_correlation_dimension(self, rr_intervals: np.ndarray) -> float:
        """Calculate correlation dimension"""
        # Simplified correlation dimension calculation
        if len(rr_intervals) < 50:
            return None
        
        try:
            # Embed the time series
            m = 5  # Embedding dimension
            embedded = np.array([rr_intervals[i:i + m] for i in range(len(rr_intervals) - m + 1)])
            
            # Calculate correlation sum for different radii
            radii = np.logspace(-2, 0, 10) * np.std(rr_intervals)
            correlation_sums = []
            
            for r in radii:
                count = 0
                n_points = len(embedded)
                
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        if np.max(np.abs(embedded[i] - embedded[j])) < r:
                            count += 1
                
                correlation_sums.append(2 * count / (n_points * (n_points - 1)))
            
            # Fit line to log-log plot
            valid_indices = [i for i, cs in enumerate(correlation_sums) if cs > 0]
            if len(valid_indices) < 3:
                return None
            
            log_radii = np.log(radii[valid_indices])
            log_corr_sums = np.log([correlation_sums[i] for i in valid_indices])
            
            correlation_dimension = np.polyfit(log_radii, log_corr_sums, 1)[0]
            
            return correlation_dimension
            
        except:
            return None
    
    def _estimate_baroreflex_sensitivity(self, rr_intervals: np.ndarray) -> float:
        """Estimate baroreflex sensitivity from HRV"""
        # Simplified BRS estimation based on HRV patterns
        if len(rr_intervals) < 10:
            return None
        
        # BRS correlates with HF power and RMSSD
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        sdnn = np.std(rr_intervals)
        
        # Empirical formula based on literature
        brs_estimate = rmssd * 0.5 + sdnn * 0.3
        
        return brs_estimate
    
    def _estimate_endothelial_function(self, ppg_signal: np.ndarray) -> float:
        """Estimate endothelial function from PPG morphology"""
        # Simplified endothelial function estimation
        try:
            # Calculate pulse wave reflection index
            peaks, _ = signal.find_peaks(ppg_signal, height=0.3)
            
            if len(peaks) < 2:
                return None
            
            # Analyze pulse wave reflection characteristics
            pulse_segments = []
            for i in range(len(peaks) - 1):
                segment = ppg_signal[peaks[i]:peaks[i + 1]]
                if len(segment) > 10:
                    pulse_segments.append(segment)
            
            if not pulse_segments:
                return None
            
            # Calculate reflection index (simplified)
            reflection_indices = []
            for segment in pulse_segments[:5]:  # Use first 5 pulses
                normalized = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
                # Find dicrotic notch approximation
                peak_idx = np.argmax(normalized)
                if peak_idx < len(normalized) - 3:
                    post_peak = normalized[peak_idx:]
                    reflection_idx = np.min(post_peak) / np.max(normalized)
                    reflection_indices.append(reflection_idx)
            
            if reflection_indices:
                # Lower reflection index suggests better endothelial function
                endothelial_index = 1.0 - np.mean(reflection_indices)
                return max(0, min(1, endothelial_index))
            
            return None
            
        except:
            return None
    
    # Empty feature dictionaries for error handling
    def _get_empty_features(self) -> Dict[str, None]:
        """Return empty feature dictionary"""
        return {key: None for key in [
            'mean_pulse_width', 'pulse_transit_time', 'enhanced_abi',
            'vlf_power', 'lf_power', 'hf_power', 'approximate_entropy'
        ]}
    
    def _get_empty_morphology_features(self) -> Dict[str, None]:
        return {key: None for key in [
            'mean_pulse_width', 'std_pulse_width', 'mean_systolic_amplitude',
            'pulse_pressure_ratio', 'pulse_regularity_index'
        ]}
    
    def _get_empty_hrv_features(self) -> Dict[str, None]:
        return {key: None for key in [
            'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'triangular_index',
            'poincare_sd1', 'poincare_sd2'
        ]}
    
    def _get_empty_frequency_features(self) -> Dict[str, None]:
        return {key: None for key in [
            'vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio',
            'normalized_lf', 'normalized_hf'
        ]}
    
    def _get_empty_nonlinear_features(self) -> Dict[str, None]:
        return {key: None for key in [
            'approximate_entropy', 'sample_entropy', 'dfa_alpha1',
            'dfa_alpha2', 'correlation_dimension'
        ]}


# Example usage and testing
if __name__ == "__main__":
    # Test the advanced feature extractor
    extractor = AdvancedPPGFeatureExtractor(sampling_rate=30.0)
    
    # Generate synthetic PPG signal for testing
    t = np.linspace(0, 30, 900)  # 30 seconds at 30 Hz
    heart_rate = 70  # BPM
    ppg_synthetic = np.sin(2 * np.pi * heart_rate / 60 * t) + 0.1 * np.random.randn(len(t))
    
    # Add user metadata
    metadata = {
        'age': 28,
        'sex': 'female',
        'cycle_phase': 'luteal',
        'fitness_level': 0.7
    }
    
    # Extract comprehensive features
    features = extractor.extract_comprehensive_features(ppg_synthetic, metadata)
    
    print("Advanced PPG Feature Extraction Results:")
    print("=" * 50)
    
    feature_categories = [
        ('Pulse Morphology', ['mean_pulse_width', 'pulse_pressure_ratio', 'pulse_regularity_index']),
        ('Pulse Transit Time', ['mean_pulse_transit_time', 'arterial_stiffness_index', 'estimated_pwv']),
        ('Advanced HRV', ['sdnn', 'rmssd', 'pnn50', 'triangular_index']),
        ('Frequency Domain', ['vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio']),
        ('Nonlinear Features', ['approximate_entropy', 'sample_entropy', 'dfa_alpha1']),
        ('Physiological Indices', ['enhanced_abi', 'baroreflex_sensitivity', 'endothelial_function_index'])
    ]
    
    for category, feature_list in feature_categories:
        print(f"\n{category}:")
        for feature in feature_list:
            value = features.get(feature)
            if value is not None:
                if isinstance(value, float):
                    print(f"  {feature}: {value:.4f}")
                else:
                    print(f"  {feature}: {value}")
            else:
                print(f"  {feature}: N/A")
    
    print(f"\nTotal features extracted: {len([v for v in features.values() if v is not None])}")