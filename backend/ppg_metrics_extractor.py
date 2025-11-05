"""
Advanced PPG Metrics Extraction System
Implements comprehensive cardiovascular metrics extraction from PPG signals
Including HR, RHR, HRV metrics, frequency domain analysis, and clinical indices
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import warnings
from typing import Dict, List, Tuple, Optional, Any

# AI risk assessment
try:
    from ppg_risk_ai import assess_cardiovascular_risk_ai
    AI_RISK_AVAILABLE = True
except ImportError:
    AI_RISK_AVAILABLE = False

# Suppress runtime warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class PPGMetricsExtractor:
    """
    Comprehensive PPG metrics extraction with clinical-grade calculations
    """
    
    def __init__(self, fs: int = 30):
        """
        Initialize the metrics extractor
        
        Args:
            fs: Sampling frequency in Hz (default: 30 for camera-based PPG)
        """
        self.fs = fs
        self.nyquist = fs / 2
        
        # Clinical thresholds based on literature
        self.clinical_thresholds = {
            'hr_normal_range': (60, 100),
            'rhr_normal_range': (50, 90),
            'rmssd_normal_range': (20, 50),  # ms
            'sdnn_normal_range': (30, 100),  # ms
            'pnn50_normal_range': (10, 40),  # %
            'lf_hf_normal_range': (0.5, 2.0),
            'artifact_threshold': 15,  # % acceptable artifact level
        }
        
        # Phase-adaptive baseline tracking
        self.baseline_history = {
            'hr': [],
            'rhr': [],
            'rmssd': [],
            'sdnn': [],
            'lf_hf': []
        }
        
        self.baseline_window_size = 20  # measurements for baseline calculation
    
    def extract_comprehensive_metrics(self, ppg_signal: np.ndarray, 
                                    processed_signal: Optional[np.ndarray] = None,
                                    quality_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract comprehensive cardiovascular metrics from PPG signal
        
        Args:
            ppg_signal: Raw PPG signal
            processed_signal: Pre-processed PPG signal (optional)
            quality_info: Signal quality information (optional)
            
        Returns:
            Dictionary containing all extracted metrics and clinical indices
        """
        
        if processed_signal is None:
            processed_signal = self._preprocess_for_analysis(ppg_signal)
        
        # 1. Beat Detection and RR Interval Extraction
        peaks, rr_intervals = self._extract_rr_intervals(processed_signal)
        
        if len(rr_intervals) < 5:
            return self._generate_empty_metrics("Insufficient beats detected")
        
        # 2. Time Domain Metrics
        time_metrics = self._calculate_time_domain_metrics(rr_intervals)
        
        # 3. Frequency Domain Metrics  
        freq_metrics = self._calculate_frequency_domain_metrics(rr_intervals)
        
        # 4. Resting Heart Rate (RHR) Estimation
        rhr_metrics = self._estimate_resting_heart_rate(time_metrics['hr'], rr_intervals)
        
        # 5. Signal Quality Assessment
        quality_metrics = self._assess_signal_quality(ppg_signal, processed_signal, peaks)
        
        # 6. Clinical Indices (ABI, CVR, CSI)
        clinical_indices = self._calculate_clinical_indices(time_metrics, freq_metrics, quality_metrics)
        
        # 7. Phase-Adaptive Baseline Calculation
        baseline_metrics = self._calculate_phase_adaptive_baselines(time_metrics, freq_metrics)
        
        # 8. Trend Detection and Risk Flags
        trend_analysis = self._detect_trends_and_flags(time_metrics, freq_metrics, baseline_metrics)
        
        # 9. AI-Based Risk Assessment (if available)
        ai_risk_analysis = self._perform_ai_risk_assessment(time_metrics, freq_metrics, quality_metrics)
        
        # Combine all metrics
        comprehensive_metrics = {
            **time_metrics,
            **freq_metrics,
            **rhr_metrics,
            **quality_metrics,
            **clinical_indices,
            **baseline_metrics,
            **trend_analysis,
            **ai_risk_analysis,
            'raw_data': {
                'rr_intervals': rr_intervals.tolist(),
                'peak_positions': peaks.tolist(),
                'signal_length': len(ppg_signal),
                'sampling_rate': self.fs
            },
            'extraction_timestamp': pd.Timestamp.now().isoformat(),
            'extraction_quality': 'high' if len(rr_intervals) >= 20 else 'moderate' if len(rr_intervals) >= 10 else 'low'
        }
        
        return comprehensive_metrics
    
    def _preprocess_for_analysis(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Preprocess PPG signal for analysis"""
        try:
            # Bandpass filter (0.5-4 Hz for PPG)
            sos = signal.butter(4, [0.5, 4.0], btype='band', fs=self.fs, output='sos')
            filtered = signal.sosfiltfilt(sos, ppg_signal)
            
            # Normalize
            filtered = (filtered - np.mean(filtered)) / np.std(filtered)
            
            return filtered
        except:
            return ppg_signal
    
    def _extract_rr_intervals(self, processed_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract RR intervals using advanced peak detection
        """
        # Adaptive peak detection parameters
        prominence = np.std(processed_signal) * 0.3
        distance = int(0.4 * self.fs)  # Minimum 400ms between peaks (150 BPM max)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            processed_signal,
            prominence=prominence,
            distance=distance,
            height=np.percentile(processed_signal, 60)
        )
        
        if len(peaks) < 2:
            return np.array([]), np.array([])
        
        # Convert peak positions to RR intervals (in milliseconds)
        rr_intervals = np.diff(peaks) * 1000 / self.fs
        
        # Filter physiologically plausible RR intervals (300-2000ms)
        valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
        rr_intervals = rr_intervals[valid_mask]
        
        # Remove outliers using IQR method
        if len(rr_intervals) > 5:
            q25, q75 = np.percentile(rr_intervals, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            rr_intervals = rr_intervals[(rr_intervals >= lower_bound) & (rr_intervals <= upper_bound)]
        
        return peaks, rr_intervals
    
    def _calculate_time_domain_metrics(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate time domain HRV metrics"""
        
        if len(rr_intervals) == 0:
            return {'hr': 0, 'rmssd': 0, 'sdnn': 0, 'pnn50': 0, 'mean_rr': 0, 'cv_rr': 0}
        
        # Basic statistics
        mean_rr = np.mean(rr_intervals)
        hr = 60000 / mean_rr if mean_rr > 0 else 0
        
        # HRV metrics
        sdnn = np.std(rr_intervals, ddof=1)
        
        # RMSSD (Root Mean Square of Successive Differences)
        if len(rr_intervals) > 1:
            successive_diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(successive_diffs**2))
        else:
            rmssd = 0
        
        # pNN50 (percentage of successive RR intervals differing by > 50ms)
        if len(rr_intervals) > 1:
            nn50_count = np.sum(np.abs(successive_diffs) > 50)
            pnn50 = (nn50_count / len(successive_diffs)) * 100
        else:
            pnn50 = 0
        
        # Coefficient of variation
        cv_rr = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        
        return {
            'hr': round(hr, 1),
            'rmssd': round(rmssd, 2),
            'sdnn': round(sdnn, 2),
            'pnn50': round(pnn50, 2),
            'mean_rr': round(mean_rr, 2),
            'cv_rr': round(cv_rr, 2)
        }
    
    def _calculate_frequency_domain_metrics(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate frequency domain HRV metrics"""
        
        if len(rr_intervals) < 10:
            return {'lf_power': 0, 'hf_power': 0, 'lf_hf_ratio': 0, 'total_power': 0, 'vlf_power': 0}
        
        try:
            # Interpolate RR intervals to create evenly spaced time series
            time_rr = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
            time_rr = time_rr - time_rr[0]  # Start from 0
            
            # Create interpolation function
            interp_func = interp1d(time_rr, rr_intervals, kind='cubic', fill_value='extrapolate')
            
            # Create evenly spaced time series (4 Hz sampling)
            fs_interp = 4.0
            time_even = np.arange(0, time_rr[-1], 1/fs_interp)
            rr_interp = interp_func(time_even)
            
            # Remove trend (detrending)
            rr_detrended = signal.detrend(rr_interp)
            
            # Apply Hanning window
            windowed = rr_detrended * np.hanning(len(rr_detrended))
            
            # Calculate power spectral density
            frequencies, psd = signal.welch(windowed, fs=fs_interp, nperseg=len(windowed)//2)
            
            # Define frequency bands (Hz)
            vlf_band = (0.0033, 0.04)  # Very Low Frequency
            lf_band = (0.04, 0.15)     # Low Frequency
            hf_band = (0.15, 0.4)      # High Frequency
            
            # Calculate power in each band
            vlf_power = self._calculate_band_power(frequencies, psd, vlf_band)
            lf_power = self._calculate_band_power(frequencies, psd, lf_band)
            hf_power = self._calculate_band_power(frequencies, psd, hf_band)
            total_power = vlf_power + lf_power + hf_power
            
            # LF/HF ratio
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
            
            return {
                'lf_power': round(lf_power, 2),
                'hf_power': round(hf_power, 2),
                'lf_hf_ratio': round(lf_hf_ratio, 3),
                'total_power': round(total_power, 2),
                'vlf_power': round(vlf_power, 2)
            }
            
        except Exception as e:
            return {'lf_power': 0, 'hf_power': 0, 'lf_hf_ratio': 0, 'total_power': 0, 'vlf_power': 0}
    
    def _calculate_band_power(self, frequencies: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
        """Calculate power in specified frequency band"""
        band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
        return np.trapz(psd[band_mask], frequencies[band_mask])
    
    def _estimate_resting_heart_rate(self, current_hr: float, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Estimate resting heart rate using moving average and percentile methods"""
        
        # Calculate RHR as 10th percentile of recent measurements
        if len(rr_intervals) >= 10:
            rr_10th = np.percentile(rr_intervals, 90)  # 90th percentile of RR = 10th percentile of HR
            rhr_percentile = 60000 / rr_10th if rr_10th > 0 else current_hr
        else:
            rhr_percentile = current_hr
        
        # Update baseline history for trend analysis
        self.baseline_history['hr'].append(current_hr)
        if len(self.baseline_history['hr']) > self.baseline_window_size:
            self.baseline_history['hr'].pop(0)
        
        # Calculate RHR as minimum of recent measurements
        if len(self.baseline_history['hr']) >= 5:
            rhr_minimum = min(self.baseline_history['hr'])
        else:
            rhr_minimum = current_hr
        
        # Final RHR estimate (average of methods)
        rhr_estimate = (rhr_percentile + rhr_minimum) / 2
        
        return {
            'rhr_estimate': round(rhr_estimate, 1),
            'rhr_percentile': round(rhr_percentile, 1),
            'rhr_minimum': round(rhr_minimum, 1),
            'hr_elevation': round(current_hr - rhr_estimate, 1)
        }
    
    def _assess_signal_quality(self, raw_signal: np.ndarray, processed_signal: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
        """Comprehensive signal quality assessment"""
        
        # SNR calculation
        signal_power = np.var(processed_signal)
        noise_estimate = np.var(raw_signal - processed_signal)
        snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 20
        
        # Peak quality assessment
        if len(peaks) > 1:
            peak_regularity = 1 - (np.std(np.diff(peaks)) / np.mean(np.diff(peaks)))
            peak_amplitude_cv = np.std(processed_signal[peaks]) / np.mean(processed_signal[peaks])
        else:
            peak_regularity = 0
            peak_amplitude_cv = 1
        
        # Artifact percentage estimation
        artifact_percentage = max(0, min(100, (1 - peak_regularity) * 100))
        
        # Overall quality score (0-100)
        quality_score = max(0, min(100, 
            (snr / 20) * 40 +  # SNR contributes 40%
            peak_regularity * 40 +  # Peak regularity contributes 40%
            (1 - min(peak_amplitude_cv, 1)) * 20  # Amplitude consistency contributes 20%
        ))
        
        return {
            'signal_quality': round(quality_score, 1),
            'snr_db': round(snr, 2),
            'peak_regularity': round(peak_regularity, 3),
            'artifact_percentage': round(artifact_percentage, 1),
            'peak_count': len(peaks),
            'peak_amplitude_cv': round(peak_amplitude_cv, 3)
        }
    
    def _calculate_clinical_indices(self, time_metrics: Dict, freq_metrics: Dict, quality_metrics: Dict) -> Dict[str, float]:
        """
        Calculate clinical indices: ABI, CVR, CSI
        ABI: Autonomic Balance Index
        CVR: Cardiovascular Risk Index  
        CSI: Cardiac Stress Index
        """
        
        # Autonomic Balance Index (ABI) - based on LF/HF ratio and RMSSD
        lf_hf = freq_metrics.get('lf_hf_ratio', 1.0)
        rmssd = time_metrics.get('rmssd', 20.0)
        
        # Normalize LF/HF ratio (ideal range 0.5-2.0)
        lf_hf_normalized = max(0, min(1, (2.0 - abs(lf_hf - 1.0)) / 2.0))
        
        # Normalize RMSSD (ideal range 20-50ms)
        rmssd_normalized = max(0, min(1, rmssd / 50.0))
        
        abi = (lf_hf_normalized * 0.6 + rmssd_normalized * 0.4) * 100
        
        # Cardiovascular Risk Index (CVR) - composite of multiple factors
        hr = time_metrics.get('hr', 70)
        sdnn = time_metrics.get('sdnn', 40)
        quality = quality_metrics.get('signal_quality', 70)
        
        # Risk factors (higher values = higher risk)
        hr_risk = max(0, (abs(hr - 70) - 10) / 30)  # Risk increases beyond ±10 from 70 BPM
        hrv_risk = max(0, (30 - sdnn) / 30)  # Risk increases as SDNN drops below 30ms
        quality_risk = max(0, (70 - quality) / 70)  # Risk increases with poor signal quality
        
        cvr = (hr_risk * 0.4 + hrv_risk * 0.4 + quality_risk * 0.2) * 100
        
        # Cardiac Stress Index (CSI) - immediate stress assessment
        pnn50 = time_metrics.get('pnn50', 15)
        hf_power = freq_metrics.get('hf_power', 100)
        
        # Stress indicators (lower parasympathetic activity = higher stress)
        pnn50_stress = max(0, (20 - pnn50) / 20)
        hf_stress = max(0, (100 - hf_power) / 100) if hf_power > 0 else 0.5
        
        csi = (pnn50_stress * 0.6 + hf_stress * 0.4) * 100
        
        return {
            'abi_score': round(abi, 1),
            'cvr_score': round(cvr, 1),
            'csi_score': round(csi, 1),
            'abi_interpretation': self._interpret_abi(abi),
            'cvr_interpretation': self._interpret_cvr(cvr),
            'csi_interpretation': self._interpret_csi(csi)
        }
    
    def _interpret_abi(self, abi: float) -> str:
        """Interpret Autonomic Balance Index"""
        if abi >= 70:
            return "Excellent autonomic balance"
        elif abi >= 50:
            return "Good autonomic balance"
        elif abi >= 30:
            return "Fair autonomic balance"
        else:
            return "Poor autonomic balance"
    
    def _interpret_cvr(self, cvr: float) -> str:
        """Interpret Cardiovascular Risk Index"""
        if cvr <= 20:
            return "Low cardiovascular risk"
        elif cvr <= 40:
            return "Moderate cardiovascular risk"
        elif cvr <= 60:
            return "Elevated cardiovascular risk"
        else:
            return "High cardiovascular risk"
    
    def _interpret_csi(self, csi: float) -> str:
        """Interpret Cardiac Stress Index"""
        if csi <= 25:
            return "Low stress level"
        elif csi <= 50:
            return "Moderate stress level"
        elif csi <= 75:
            return "Elevated stress level"
        else:
            return "High stress level"
    
    def _calculate_phase_adaptive_baselines(self, time_metrics: Dict, freq_metrics: Dict) -> Dict[str, Any]:
        """Calculate phase-adaptive baselines for trend detection"""
        
        # Update baseline history
        metrics_to_track = ['rmssd', 'sdnn', 'lf_hf_ratio']
        current_values = {
            'rmssd': time_metrics.get('rmssd', 0),
            'sdnn': time_metrics.get('sdnn', 0),
            'lf_hf_ratio': freq_metrics.get('lf_hf_ratio', 0)
        }
        
        baselines = {}
        deviations = {}
        
        for metric in metrics_to_track:
            # Add current value to history
            if metric not in self.baseline_history:
                self.baseline_history[metric] = []
            
            self.baseline_history[metric].append(current_values[metric])
            
            # Keep only recent measurements
            if len(self.baseline_history[metric]) > self.baseline_window_size:
                self.baseline_history[metric].pop(0)
            
            # Calculate baseline (median of recent measurements)
            if len(self.baseline_history[metric]) >= 3:
                baseline = np.median(self.baseline_history[metric])
                std_dev = np.std(self.baseline_history[metric])
                
                # Calculate deviation from baseline
                current_val = current_values[metric]
                deviation = (current_val - baseline) / std_dev if std_dev > 0 else 0
                
                baselines[f'{metric}_baseline'] = round(baseline, 3)
                baselines[f'{metric}_std'] = round(std_dev, 3)
                deviations[f'{metric}_deviation'] = round(deviation, 2)
            else:
                baselines[f'{metric}_baseline'] = current_values[metric]
                baselines[f'{metric}_std'] = 0
                deviations[f'{metric}_deviation'] = 0
        
        return {
            **baselines,
            **deviations,
            'baseline_measurements': len(self.baseline_history.get('rmssd', [])),
            'baseline_stability': self._assess_baseline_stability()
        }
    
    def _assess_baseline_stability(self) -> str:
        """Assess stability of baseline measurements"""
        if len(self.baseline_history.get('rmssd', [])) < 5:
            return "Establishing baseline"
        elif len(self.baseline_history.get('rmssd', [])) < 10:
            return "Baseline developing"
        else:
            # Check coefficient of variation
            rmssd_values = self.baseline_history.get('rmssd', [])
            if len(rmssd_values) > 1:
                cv = np.std(rmssd_values) / np.mean(rmssd_values)
                if cv < 0.2:
                    return "Stable baseline"
                elif cv < 0.4:
                    return "Moderately stable baseline"
                else:
                    return "Variable baseline"
            else:
                return "Insufficient data"
    
    def _detect_trends_and_flags(self, time_metrics: Dict, freq_metrics: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Detect trends and generate risk flags based on literature-derived thresholds"""
        
        flags = []
        risk_level = "Low"
        trend_indicators = {}
        
        # Extract current values
        hr = time_metrics.get('hr', 70)
        rmssd = time_metrics.get('rmssd', 25)
        sdnn = time_metrics.get('sdnn', 40)
        lf_hf = freq_metrics.get('lf_hf_ratio', 1.0)
        
        # Extract deviations from baseline
        rmssd_dev = baseline_metrics.get('rmssd_deviation', 0)
        sdnn_dev = baseline_metrics.get('sdnn_deviation', 0)
        lf_hf_dev = baseline_metrics.get('lf_hf_ratio_deviation', 0)
        
        # Flag 1: Bradycardia/Tachycardia
        if hr < 50:
            flags.append("Bradycardia detected (HR < 50 BPM)")
            risk_level = "High"
        elif hr > 120:
            flags.append("Tachycardia detected (HR > 120 BPM)")
            risk_level = "High"
        elif hr < 60 or hr > 100:
            flags.append("Heart rate outside normal range")
            risk_level = max(risk_level, "Moderate")
        
        # Flag 2: Low HRV (cardiovascular risk indicator)
        if rmssd < 15:
            flags.append("Very low HRV - increased cardiovascular risk")
            risk_level = "High"
        elif rmssd < 20:
            flags.append("Low HRV detected")
            risk_level = max(risk_level, "Moderate")
        
        # Flag 3: Autonomic imbalance
        if lf_hf > 4.0:
            flags.append("Sympathetic dominance - high stress/inflammation")
            risk_level = max(risk_level, "Moderate")
        elif lf_hf < 0.2:
            flags.append("Parasympathetic dominance - possible overtraining")
            risk_level = max(risk_level, "Moderate")
        
        # Flag 4: Significant baseline deviations (trend detection)
        if abs(rmssd_dev) > 2.0:
            direction = "increase" if rmssd_dev > 0 else "decrease"
            flags.append(f"Significant RMSSD {direction} from baseline")
            risk_level = max(risk_level, "Moderate")
        
        if abs(sdnn_dev) > 2.0:
            direction = "increase" if sdnn_dev > 0 else "decrease"
            flags.append(f"Significant SDNN {direction} from baseline")
            risk_level = max(risk_level, "Moderate")
        
        # Flag 5: Literature-based risk patterns
        # Pattern 1: Low HRV + High HR (metabolic syndrome risk)
        if rmssd < 25 and hr > 80:
            flags.append("Pattern: Low HRV + elevated HR - metabolic risk")
            risk_level = "High"
        
        # Pattern 2: Very low SDNN (all-cause mortality risk)
        if sdnn < 20:
            flags.append("Very low SDNN - increased mortality risk (literature)")
            risk_level = "High"
        
        # Pattern 3: Extreme LF/HF imbalance
        if lf_hf > 6.0 or lf_hf < 0.1:
            flags.append("Extreme autonomic imbalance detected")
            risk_level = "High"
        
        # Trend indicators for visualization
        trend_indicators = {
            'hr_trend': self._classify_trend(hr, 70, 10),
            'rmssd_trend': self._classify_trend(rmssd, 35, 10),
            'sdnn_trend': self._classify_trend(sdnn, 50, 15),
            'lf_hf_trend': self._classify_trend(lf_hf, 1.0, 0.5)
        }
        
        return {
            'risk_flags': flags,
            'risk_level': risk_level,
            'flag_count': len(flags),
            'trend_indicators': trend_indicators,
            'clinical_recommendations': self._generate_clinical_recommendations(flags, time_metrics, freq_metrics)
        }
    
    def _classify_trend(self, current_value: float, normal_value: float, threshold: float) -> str:
        """Classify trend as increasing, decreasing, or stable"""
        deviation = abs(current_value - normal_value)
        if current_value > normal_value + threshold:
            return "elevated"
        elif current_value < normal_value - threshold:
            return "reduced"
        else:
            return "normal"
    
    def _generate_clinical_recommendations(self, flags: List[str], time_metrics: Dict, freq_metrics: Dict) -> List[str]:
        """Generate clinical recommendations based on detected flags"""
        
        recommendations = []
        
        hr = time_metrics.get('hr', 70)
        rmssd = time_metrics.get('rmssd', 25)
        lf_hf = freq_metrics.get('lf_hf_ratio', 1.0)
        
        # HR-based recommendations
        if hr > 100:
            recommendations.append("Consider relaxation techniques or medical evaluation for tachycardia")
        elif hr < 60:
            recommendations.append("Monitor for symptoms; consider medical evaluation if symptomatic")
        
        # HRV-based recommendations
        if rmssd < 20:
            recommendations.append("Low HRV: Consider stress management, regular exercise, adequate sleep")
        
        # Autonomic balance recommendations
        if lf_hf > 3.0:
            recommendations.append("High stress indicated: Practice meditation, deep breathing exercises")
        elif lf_hf < 0.3:
            recommendations.append("Consider reducing training intensity if applicable")
        
        # General recommendations if no specific flags
        if not recommendations:
            recommendations.append("Cardiovascular metrics within normal ranges - maintain healthy lifestyle")
        
        # Always add measurement quality recommendation
        recommendations.append("Ensure steady posture and minimal movement during measurements")
        
        return recommendations
    
    def _perform_ai_risk_assessment(self, time_metrics: Dict, freq_metrics: Dict, 
                                   quality_metrics: Dict) -> Dict[str, Any]:
        """
        Perform AI-based cardiovascular risk assessment
        """
        
        if not AI_RISK_AVAILABLE:
            return {
                'ai_assessment_available': False,
                'ai_risk_level': 'Unknown',
                'ai_recommendations': ['AI risk assessment not available'],
                'literature_patterns': [],
                'ai_confidence': 0
            }
        
        try:
            # Prepare metrics for AI assessment
            ai_input_metrics = {
                **time_metrics,
                **freq_metrics,
                **quality_metrics
            }
            
            # Perform AI assessment
            ai_results = assess_cardiovascular_risk_ai(ai_input_metrics)
            
            # Extract key results
            ai_assessment = ai_results.get('ai_risk_assessment', {})
            pattern_detection = ai_results.get('pattern_detection', [])
            literature_flags = ai_results.get('literature_flags', [])
            ai_recommendations = ai_results.get('ai_recommendations', [])
            
            return {
                'ai_assessment_available': True,
                'ai_risk_level': ai_assessment.get('risk_level', 'Unknown'),
                'ai_risk_probabilities': ai_assessment.get('risk_probabilities', {}),
                'ai_trend_score': ai_assessment.get('trend_score', 0),
                'ai_confidence': ai_assessment.get('confidence', 0),
                'literature_patterns': pattern_detection,
                'literature_flags': literature_flags,
                'ai_recommendations': ai_recommendations,
                'pattern_count': len(pattern_detection),
                'literature_flag_count': len(literature_flags)
            }
            
        except Exception as e:
            return {
                'ai_assessment_available': False,
                'ai_error': str(e),
                'ai_risk_level': 'Error',
                'ai_recommendations': ['AI assessment failed - using traditional methods'],
                'literature_patterns': [],
                'ai_confidence': 0
            }
    
    def _generate_empty_metrics(self, reason: str) -> Dict[str, Any]:
        """Generate empty metrics dictionary when extraction fails"""
        return {
            'extraction_status': 'failed',
            'failure_reason': reason,
            'hr': 0,
            'rmssd': 0,
            'sdnn': 0,
            'pnn50': 0,
            'lf_hf_ratio': 0,
            'signal_quality': 0,
            'risk_flags': ['Insufficient signal quality for analysis'],
            'risk_level': 'Unknown',
            'clinical_recommendations': ['Improve signal quality and repeat measurement']
        }

def extract_ppg_metrics(ppg_signal: np.ndarray, fs: int = 30, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for extracting PPG metrics
    
    Args:
        ppg_signal: PPG signal array
        fs: Sampling frequency
        **kwargs: Additional arguments for the extractor
        
    Returns:
        Comprehensive metrics dictionary
    """
    extractor = PPGMetricsExtractor(fs=fs)
    return extractor.extract_comprehensive_metrics(ppg_signal, **kwargs)