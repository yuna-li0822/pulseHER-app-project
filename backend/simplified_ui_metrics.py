"""
PulseHER UX Enhancement - 10/10 Heart Resilience Score & Simplified Interface
Transforms complex metrics into intuitive 0-100 scores with explanations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class HeartResilienceCalculator:
    """
    Converts complex HRV metrics into intuitive 0-100 Heart Resilience Score
    """
    
    def __init__(self):
        # Age-based population norms for RMSSD (ms) - research-derived
        self.age_norms = {
            '18-25': {'rmssd_mean': 42, 'rmssd_std': 15, 'hr_mean': 70, 'hr_std': 12},
            '26-35': {'rmssd_mean': 35, 'rmssd_std': 13, 'hr_mean': 72, 'hr_std': 11},
            '36-45': {'rmssd_mean': 28, 'rmssd_std': 11, 'hr_mean': 74, 'hr_std': 10},
            '46-55': {'rmssd_mean': 23, 'rmssd_std': 9, 'hr_mean': 76, 'hr_std': 11},
            '56-65': {'rmssd_mean': 19, 'rmssd_std': 8, 'hr_mean': 78, 'hr_std': 12}
        }
        
        # Score component weights
        self.weights = {
            'rmssd': 0.4,      # Primary HRV metric
            'hr_recovery': 0.3, # Resting HR relative to age
            'consistency': 0.2, # Measurement quality/consistency
            'trend': 0.1       # Recent trend direction
        }
    
    def calculate_resilience_score(self, metrics: Dict, user_profile: Dict, 
                                 recent_history: List[Dict] = None) -> Dict:
        """
        Calculate comprehensive Heart Resilience Score (0-100)
        """
        age = self._calculate_age(user_profile.get('date_of_birth'))
        sex = user_profile.get('sex_at_birth', 'female')
        
        # Get age-appropriate norms
        age_group = self._get_age_group(age)
        norms = self.age_norms.get(age_group, self.age_norms['26-35'])
        
        # Component 1: HRV Score (RMSSD-based)
        rmssd_score = self._calculate_rmssd_score(
            metrics.get('rmssd_ms', 0), norms, sex
        )
        
        # Component 2: Heart Rate Recovery Score
        hr_score = self._calculate_hr_score(
            metrics.get('mean_hr_bpm', 0), norms, sex
        )
        
        # Component 3: Measurement Consistency Score
        consistency_score = self._calculate_consistency_score(
            metrics, recent_history
        )
        
        # Component 4: Trend Score
        trend_score = self._calculate_trend_score(recent_history)
        
        # Weighted composite score
        composite_score = (
            self.weights['rmssd'] * rmssd_score +
            self.weights['hr_recovery'] * hr_score +
            self.weights['consistency'] * consistency_score +
            self.weights['trend'] * trend_score
        )
        
        # Ensure score is 0-100
        final_score = max(0, min(100, composite_score))
        
        return {
            'heart_resilience_score': round(final_score, 1),
            'grade': self._get_score_grade(final_score),
            'components': {
                'hrv_component': round(rmssd_score, 1),
                'hr_component': round(hr_score, 1),
                'consistency_component': round(consistency_score, 1),
                'trend_component': round(trend_score, 1)
            },
            'explanation': self._generate_score_explanation(final_score, age_group),
            'percentile': self._calculate_percentile(final_score, age_group, sex),
            'improvement_tips': self._get_improvement_tips(final_score, metrics)
        }
    
    def _calculate_age(self, date_of_birth: str) -> int:
        """Calculate age from date of birth"""
        try:
            if not date_of_birth:
                return 30  # Default age
            
            birth_date = datetime.fromisoformat(date_of_birth.replace('Z', ''))
            today = datetime.now()
            age = today.year - birth_date.year
            
            if today.month < birth_date.month or \
               (today.month == birth_date.month and today.day < birth_date.day):
                age -= 1
                
            return max(18, min(65, age))  # Clamp to valid range
        except:
            return 30
    
    def _get_age_group(self, age: int) -> str:
        """Get age group for norms lookup"""
        if age < 26:
            return '18-25'
        elif age < 36:
            return '26-35'
        elif age < 46:
            return '36-45'
        elif age < 56:
            return '46-55'
        else:
            return '56-65'
    
    def _calculate_rmssd_score(self, rmssd: float, norms: Dict, sex: str) -> float:
        """
        Calculate HRV component score (0-100) based on RMSSD
        """
        if rmssd <= 0:
            return 0
        
        # Adjust norms slightly for sex differences (women typically higher)
        sex_multiplier = 1.1 if sex == 'female' else 1.0
        adjusted_mean = norms['rmssd_mean'] * sex_multiplier
        
        # Convert to z-score and then to 0-100 scale
        z_score = (rmssd - adjusted_mean) / norms['rmssd_std']
        
        # Transform z-score to 0-100 scale (mean = 50, std = 15)
        score = 50 + (z_score * 15)
        
        return max(0, min(100, score))
    
    def _calculate_hr_score(self, hr: float, norms: Dict, sex: str) -> float:
        """
        Calculate HR component score (0-100) - lower HR is better
        """
        if hr <= 0:
            return 0
        
        # Adjust norms for sex (women typically higher by ~3-5 bpm)
        sex_adjustment = 4 if sex == 'female' else 0
        adjusted_mean = norms['hr_mean'] + sex_adjustment
        
        # Lower HR is better, so invert the z-score
        z_score = -(hr - adjusted_mean) / norms['hr_std']
        
        # Transform to 0-100 scale
        score = 50 + (z_score * 15)
        
        return max(0, min(100, score))
    
    def _calculate_consistency_score(self, metrics: Dict, recent_history: List[Dict]) -> float:
        """
        Calculate measurement consistency/quality score
        """
        # Start with current session quality
        artifact_pct = metrics.get('artifact_percentage', 0)
        quality_score = max(0, 100 - (artifact_pct * 2))  # Penalize artifacts
        
        # Recording length bonus
        duration = metrics.get('recording_duration', 60)
        if duration >= 300:  # 5 minutes
            quality_score *= 1.1
        elif duration < 30:  # Less than 30 seconds
            quality_score *= 0.8
        
        # Consistency bonus from history
        if recent_history and len(recent_history) >= 3:
            hr_values = [h['metrics'].get('mean_hr_bpm') for h in recent_history[-7:] 
                        if h['metrics'].get('mean_hr_bpm')]
            
            if len(hr_values) >= 3:
                cv = np.std(hr_values) / np.mean(hr_values)  # Coefficient of variation
                consistency_bonus = max(0, 20 * (1 - cv))  # Less variation = higher score
                quality_score = min(100, quality_score + consistency_bonus)
        
        return max(0, min(100, quality_score))
    
    def _calculate_trend_score(self, recent_history: List[Dict]) -> float:
        """
        Calculate trend component - improving trends get bonus points
        """
        if not recent_history or len(recent_history) < 5:
            return 50  # Neutral score for insufficient data
        
        # Get recent RMSSD values
        rmssd_values = []
        timestamps = []
        
        for session in recent_history[-14:]:  # Last 14 sessions
            if 'rmssd_ms' in session['metrics']:
                rmssd_values.append(session['metrics']['rmssd_ms'])
                timestamps.append(session['timestamp'])
        
        if len(rmssd_values) < 5:
            return 50
        
        # Calculate trend using linear regression
        x = np.arange(len(rmssd_values))
        slope, _ = np.polyfit(x, rmssd_values, 1)
        
        # Convert slope to score (positive slope = improving HRV = higher score)
        if slope > 0.5:  # Improving
            return min(100, 70 + slope * 10)
        elif slope < -0.5:  # Declining  
            return max(0, 30 + slope * 10)
        else:  # Stable
            return 50
    
    def _get_score_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 85:
            return 'A'
        elif score >= 75:
            return 'B'
        elif score >= 65:
            return 'C'
        elif score >= 55:
            return 'D'
        else:
            return 'F'
    
    def _generate_score_explanation(self, score: float, age_group: str) -> str:
        """Generate user-friendly explanation of the score"""
        if score >= 85:
            return f"Excellent heart resilience for your age group ({age_group}). Your heart shows strong adaptability and recovery capacity."
        elif score >= 75:
            return f"Good heart resilience. Above average for your age group with room for optimization."
        elif score >= 65:
            return f"Moderate heart resilience. Average for your age group - consider lifestyle improvements."
        elif score >= 55:
            return f"Below average heart resilience. Focus on stress management and recovery strategies."
        else:
            return f"Low heart resilience detected. Consider consulting a healthcare provider and prioritizing recovery."
    
    def _calculate_percentile(self, score: float, age_group: str, sex: str) -> int:
        """Estimate percentile rank for the score"""
        # Simplified percentile calculation (in real app, use population data)
        if score >= 85:
            return np.random.randint(85, 96)
        elif score >= 75:
            return np.random.randint(70, 85)
        elif score >= 65:
            return np.random.randint(45, 70)
        elif score >= 55:
            return np.random.randint(25, 45)
        else:
            return np.random.randint(5, 25)
    
    def _get_improvement_tips(self, score: float, metrics: Dict) -> List[str]:
        """Get personalized tips for improving heart resilience"""
        tips = []
        
        if score < 75:
            tips.extend([
                "Aim for 7-9 hours of quality sleep nightly",
                "Practice daily stress management (meditation, deep breathing)",
                "Include moderate cardio exercise 3-4 times per week"
            ])
        
        if score < 65:
            tips.extend([
                "Limit alcohol consumption",
                "Reduce caffeine, especially after 2 PM",
                "Consider yoga or tai chi for stress reduction"
            ])
        
        if score < 55:
            tips.extend([
                "Consult healthcare provider for comprehensive evaluation",
                "Focus on recovery before increasing exercise intensity",
                "Consider professional stress management counseling"
            ])
        
        # Always include measurement tips
        artifact_pct = metrics.get('artifact_percentage', 0)
        if artifact_pct > 15:
            tips.append("Improve measurement quality by staying still and ensuring good camera contact")
        
        return tips[:4]  # Limit to 4 tips

class SimplifiedUIMetrics:
    """
    Provides simplified, user-friendly versions of complex metrics
    """
    
    def __init__(self):
        self.resilience_calc = HeartResilienceCalculator()
    
    def get_simplified_dashboard(self, session_data: Dict, user_profile: Dict, 
                               recent_history: List[Dict] = None) -> Dict:
        """
        Create simplified dashboard with Heart Resilience Score as primary metric
        """
        metrics = session_data.get('metrics', {})
        
        # Calculate Heart Resilience Score
        resilience = self.resilience_calc.calculate_resilience_score(
            metrics, user_profile, recent_history
        )
        
        # Simplified secondary metrics
        simplified = {
            'heart_resilience': {
                'score': resilience['heart_resilience_score'],
                'grade': resilience['grade'],
                'explanation': resilience['explanation'],
                'percentile': resilience['percentile'],
                'trend_icon': self._get_trend_icon(resilience['components']['trend_component']),
                'improvement_tips': resilience['improvement_tips']
            },
            'vital_signs': {
                'resting_heart_rate': {
                    'value': metrics.get('mean_hr_bpm', 0),
                    'unit': 'BPM',
                    'status': self._get_hr_status(metrics.get('mean_hr_bpm', 0)),
                    'explanation': "Your average heart rate during rest"
                },
                'measurement_quality': {
                    'score': self._calculate_quality_score(metrics),
                    'rating': self._get_quality_rating(metrics),
                    'explanation': "How reliable this measurement is"
                }
            },
            'advanced_metrics': {
                'show_details': False,  # Hidden by default
                'rmssd': {
                    'value': metrics.get('rmssd_ms', 0),
                    'unit': 'ms',
                    'explanation': "Root Mean Square of Successive Differences - measures short-term heart rate variability"
                },
                'sdnn': {
                    'value': metrics.get('sdnn_ms', 0),
                    'unit': 'ms', 
                    'explanation': "Standard Deviation of NN intervals - measures overall heart rate variability"
                },
                'stress_balance': {
                    'lf_hf_ratio': metrics.get('lf_hf_ratio', 0),
                    'explanation': "Balance between sympathetic (stress) and parasympathetic (rest) nervous systems"
                }
            },
            'cycle_awareness': {
                'current_phase': session_data.get('cycle_phase', 'unknown'),
                'expected_changes': self._get_expected_phase_changes(session_data.get('cycle_phase')),
                'phase_impact': self._get_phase_impact_message(metrics, session_data.get('cycle_phase'))
            }
        }
        
        return simplified
    
    def _get_trend_icon(self, trend_score: float) -> str:
        """Get trend direction icon"""
        if trend_score > 60:
            return "↗️"  # Improving
        elif trend_score < 40:
            return "↘️"  # Declining
        else:
            return "→"   # Stable
    
    def _get_hr_status(self, hr: float) -> str:
        """Get heart rate status"""
        if hr == 0:
            return "unknown"
        elif hr < 50:
            return "low"
        elif hr > 100:
            return "elevated"
        else:
            return "normal"
    
    def _calculate_quality_score(self, metrics: Dict) -> int:
        """Calculate quality score 0-100"""
        artifact_pct = metrics.get('artifact_percentage', 0)
        duration = metrics.get('recording_duration', 60)
        
        base_score = max(0, 100 - (artifact_pct * 2))
        
        # Duration bonus/penalty
        if duration >= 60:
            base_score *= 1.1
        elif duration < 30:
            base_score *= 0.8
        
        return min(100, int(base_score))
    
    def _get_quality_rating(self, metrics: Dict) -> str:
        """Get quality rating text"""
        score = self._calculate_quality_score(metrics)
        
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        else:
            return "Poor"
    
    def _get_expected_phase_changes(self, cycle_phase: str) -> Dict:
        """Get expected changes for cycle phase"""
        phase_changes = {
            'menstrual': {
                'hr_change': "May be slightly elevated",
                'hrv_change': "Often decreased due to inflammation",
                'energy': "Lower energy is normal"
            },
            'follicular': {
                'hr_change': "Generally stable and lower",
                'hrv_change': "Typically improving",
                'energy': "Energy levels increasing"
            },
            'ovulation': {
                'hr_change': "May peak around ovulation",
                'hrv_change': "Can fluctuate with hormone surge",
                'energy': "Peak energy and mood"
            },
            'luteal': {
                'hr_change': "Often elevated in late luteal phase",
                'hrv_change': "May decrease before next cycle",
                'energy': "Energy may decline, PMS possible"
            }
        }
        
        return phase_changes.get(cycle_phase, {
            'hr_change': "Track your patterns",
            'hrv_change': "Individual variation is normal", 
            'energy': "Listen to your body"
        })
    
    def _get_phase_impact_message(self, metrics: Dict, cycle_phase: str) -> str:
        """Get phase-specific interpretation of current metrics"""
        if not cycle_phase or cycle_phase == 'unknown':
            return "Enable cycle tracking for personalized insights"
        
        hr = metrics.get('mean_hr_bpm', 0)
        rmssd = metrics.get('rmssd_ms', 0)
        
        if cycle_phase == 'luteal':
            if hr > 75:
                return "Elevated HR is common in luteal phase - normal hormonal response"
            elif rmssd < 25:
                return "Lower HRV in luteal phase is typical - extra self-care recommended"
        elif cycle_phase == 'follicular':
            if rmssd > 35:
                return "Great HRV for follicular phase - excellent recovery capacity"
        elif cycle_phase == 'ovulation':
            return "Metrics may fluctuate during ovulation - temporary hormone surge"
        elif cycle_phase == 'menstrual':
            return "Focus on rest and recovery during menstruation"
        
        return f"Tracking your {cycle_phase} phase patterns"

# Global instances
resilience_calculator = HeartResilienceCalculator()
simplified_ui = SimplifiedUIMetrics()