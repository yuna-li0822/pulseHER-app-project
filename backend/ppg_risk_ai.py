"""
AI-Based Trend Detection and Risk Assessment for PPG Metrics
Implements machine learning models for detecting cardiovascular risk patterns
based on literature-derived data and clinical thresholds
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

class PPGRiskAssessmentAI:
    """
    AI-based risk assessment system for PPG-derived cardiovascular metrics
    Uses literature-derived patterns and clinical thresholds
    """
    
    def __init__(self, use_real_dataset=False, dataset_path=None):
        """Initialize the AI risk assessment system
        
        Args:
            use_real_dataset (bool): Whether to use real literature-derived dataset
            dataset_path (str): Path to the real dataset CSV file
        """
        
        # Model components
        self.risk_classifier = None
        self.trend_detector = None
        self.use_real_dataset = use_real_dataset
        self.dataset_path = dataset_path
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # Clinical thresholds from cardiovascular literature
        self.clinical_thresholds = {
            # Heart Rate Variability (Malik et al., 1996; Task Force, 1996)
            'rmssd_very_low': 15,      # High mortality risk
            'rmssd_low': 20,           # Moderate risk
            'rmssd_normal': 35,        # Normal range
            'sdnn_very_low': 20,       # High mortality risk
            'sdnn_low': 30,            # Moderate risk
            'sdnn_normal': 50,         # Normal range
            
            # Heart Rate (AHA/ACC guidelines)
            'hr_bradycardia': 50,      # Severe bradycardia
            'hr_low_normal': 60,       # Lower normal limit
            'hr_high_normal': 100,     # Upper normal limit
            'hr_tachycardia': 120,     # Moderate tachycardia
            
            # Autonomic Balance (Pumprla et al., 2002)
            'lf_hf_sympathetic': 4.0,  # Sympathetic dominance
            'lf_hf_balanced_high': 2.0, # Upper balanced range
            'lf_hf_balanced_low': 0.5,  # Lower balanced range
            'lf_hf_parasympathetic': 0.2, # Parasympathetic dominance
            
            # Literature-based risk patterns
            'metabolic_syndrome_hr': 80,   # HR > 80 with low HRV = metabolic risk
            'arrhythmia_rr_cv': 0.15,     # RR coefficient of variation > 15%
            'sudden_death_sdnn': 25,      # SDNN < 25ms = sudden death risk
        }
        
        # Risk pattern definitions based on literature
        self.risk_patterns = [
            {
                'name': 'MetabolicSyndromeRisk',
                'conditions': ['hr > 80', 'rmssd < 25', 'bmi_estimated > 25'],
                'risk_level': 'High',
                'reference': 'Thayer et al., 2010; Liao et al., 1997'
            },
            {
                'name': 'SuddenCardiacDeathRisk', 
                'conditions': ['sdnn < 25', 'rmssd < 15'],
                'risk_level': 'High',
                'reference': 'Kleiger et al., 1987'
            },
            {
                'name': 'AutonomicDysfunction',
                'conditions': ['lf_hf > 4.0', 'rmssd < 20'],
                'risk_level': 'Moderate',
                'reference': 'Pumprla et al., 2002'
            },
            {
                'name': 'HypertensionRisk',
                'conditions': ['hr > 85', 'lf_hf > 2.5', 'age_estimated > 40'],
                'risk_level': 'Moderate', 
                'reference': 'Schroeder et al., 2003'
            },
            {
                'name': 'StressOverload',
                'conditions': ['lf_hf > 6.0', 'rmssd < 30'],
                'risk_level': 'Moderate',
                'reference': 'Pagani et al., 1986'
            }
        ]
        
        self.is_trained = False
        
    def load_real_dataset(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load real literature-derived dataset when available
        
        Returns:
            Tuple of (features, labels) if dataset exists, (None, None) otherwise
        """
        if not self.use_real_dataset or not self.dataset_path:
            print("[INFO] No real dataset specified, using synthetic data")
            return None, None
            
        if not os.path.exists(self.dataset_path):
            print(f"[INFO] Real dataset not found at {self.dataset_path}, using synthetic data")
            return None, None
            
        try:
            print(f"[INFO] Loading real literature-derived dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            
            # Expected columns (you can modify these based on your actual dataset)
            feature_columns = ['hr', 'rmssd', 'sdnn', 'lf_hf', 'age', 'gender', 'fitness_level']
            target_column = 'risk_level'  # or whatever your target column is named
            
            # Validate required columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                print(f"[WARN] Missing columns in dataset: {missing_cols}")
                return None, None
                
            if target_column not in df.columns:
                print(f"[WARN] Target column '{target_column}' not found in dataset")
                return None, None
                
            features = df[feature_columns].values
            labels = df[target_column].values
            
            print(f"[SUCCESS] Loaded real dataset: {len(features)} samples, {len(feature_columns)} features")
            return features, labels
            
        except Exception as e:
            print(f"[ERROR] Failed to load real dataset: {e}")
            return None, None
        
    def generate_synthetic_training_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data based on literature-derived patterns
        """
        np.random.seed(42)  # For reproducibility
        
        # Feature generation based on physiological ranges
        data = []
        labels = []
        
        for i in range(n_samples):
            # Generate base physiological parameters
            age = np.random.uniform(20, 80)
            gender = np.random.choice([0, 1])  # 0=female, 1=male
            fitness_level = np.random.uniform(0, 1)  # 0=poor, 1=excellent
            
            # Generate correlated cardiovascular metrics
            # Heart rate influenced by age and fitness
            hr_base = 70 + (age - 50) * 0.2 - fitness_level * 15
            hr = max(45, min(140, hr_base + np.random.normal(0, 8)))
            
            # HRV metrics (inversely correlated with age and HR)
            rmssd_base = 45 - (age - 30) * 0.4 - (hr - 70) * 0.15 + fitness_level * 10
            rmssd = max(5, min(80, rmssd_base + np.random.normal(0, 8)))
            
            sdnn_base = 50 - (age - 30) * 0.3 - (hr - 70) * 0.1 + fitness_level * 8  
            sdnn = max(10, min(100, sdnn_base + np.random.normal(0, 6)))
            
            # pNN50 related to parasympathetic activity
            pnn50_base = 25 - (age - 30) * 0.3 + fitness_level * 15 - (hr - 70) * 0.2
            pnn50 = max(0, min(50, pnn50_base + np.random.normal(0, 5)))
            
            # LF/HF ratio (stress and autonomic balance)
            stress_factor = np.random.uniform(0, 1)
            lf_hf_base = 1.0 + stress_factor * 3.0 + (hr - 70) * 0.02
            lf_hf = max(0.1, min(8.0, lf_hf_base + np.random.normal(0, 0.5)))
            
            # Signal quality factors
            signal_quality = np.random.uniform(60, 98)
            artifact_pct = (100 - signal_quality) / 100 * 20
            
            # Create feature vector
            features = [hr, rmssd, sdnn, pnn50, lf_hf, signal_quality, 
                       age, gender, fitness_level, artifact_pct]
            
            # Determine risk label based on literature patterns
            risk_score = self._calculate_literature_risk_score(
                hr, rmssd, sdnn, pnn50, lf_hf, age, signal_quality
            )
            
            # Convert continuous risk score to categorical labels
            if risk_score < 0.3:
                risk_label = 0  # Low risk
            elif risk_score < 0.6:
                risk_label = 1  # Moderate risk
            else:
                risk_label = 2  # High risk
            
            data.append(features)
            labels.append(risk_label)
        
        return np.array(data), np.array(labels)
    
    def _calculate_literature_risk_score(self, hr: float, rmssd: float, sdnn: float, 
                                       pnn50: float, lf_hf: float, age: float, 
                                       signal_quality: float) -> float:
        """
        Calculate risk score based on literature-derived thresholds
        """
        risk_factors = []
        
        # Age risk (non-modifiable)
        if age > 65:
            risk_factors.append(0.3)
        elif age > 50:
            risk_factors.append(0.15)
        
        # Heart rate risk
        if hr < 50 or hr > 120:
            risk_factors.append(0.4)  # Severe bradycardia/tachycardia
        elif hr < 60 or hr > 100:
            risk_factors.append(0.2)  # Mild abnormality
        
        # HRV risk (Kleiger et al., 1987; Malik et al., 1996)
        if sdnn < 25:
            risk_factors.append(0.5)  # High mortality risk
        elif sdnn < 35:
            risk_factors.append(0.3)  # Moderate risk
        
        if rmssd < 15:
            risk_factors.append(0.4)  # Autonomic dysfunction
        elif rmssd < 25:
            risk_factors.append(0.2)  # Reduced parasympathetic activity
        
        # Autonomic balance risk (Pumprla et al., 2002)
        if lf_hf > 4.0:
            risk_factors.append(0.3)  # Sympathetic dominance
        elif lf_hf < 0.3:
            risk_factors.append(0.2)  # Possible overtraining
        
        # Combined risk patterns
        if hr > 80 and rmssd < 25:  # Metabolic syndrome pattern
            risk_factors.append(0.35)
            
        if lf_hf > 6.0 and rmssd < 20:  # Severe stress pattern
            risk_factors.append(0.4)
            
        # Signal quality impact
        if signal_quality < 70:
            risk_factors.append(0.1)  # Uncertainty due to poor signal
        
        # Calculate final risk score (max of individual risks, capped at 1.0)
        final_risk = min(1.0, max(risk_factors) if risk_factors else 0.1)
        
        return final_risk
    
    def train_models(self) -> Dict[str, float]:
        """
        Train AI models for risk assessment and trend detection
        Uses real literature-derived dataset if available, otherwise synthetic data
        """
        print("Training AI models for cardiovascular risk assessment...")
        
        # Try to load real dataset first
        X, y = self.load_real_dataset()
        
        # Fallback to synthetic data if real dataset not available
        if X is None or y is None:
            print("Using synthetic training data (literature-derived patterns)")
            X, y = self.generate_synthetic_training_data(n_samples=8000)
        else:
            print("Using REAL literature-derived dataset for training!")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train risk classification model
        self.risk_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.risk_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.risk_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Train trend detection model (regression for continuous risk scores)
        trend_y = np.array([self._calculate_literature_risk_score(*row[:7]) for row in X_train])
        
        self.trend_detector = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        
        self.trend_detector.fit(X_train_scaled, trend_y)
        
        # Evaluate trend detection
        trend_pred = self.trend_detector.predict(X_test_scaled)
        trend_y_test = np.array([self._calculate_literature_risk_score(*row[:7]) for row in X_test])
        trend_mae = np.mean(np.abs(trend_pred - trend_y_test))
        
        self.is_trained = True
        
        print(f"Risk Classification Accuracy: {accuracy:.3f}")
        print(f"Trend Detection MAE: {trend_mae:.3f}")
        
        return {
            'classification_accuracy': accuracy,
            'trend_detection_mae': trend_mae,
            'feature_importance': dict(zip(
                ['HR', 'RMSSD', 'SDNN', 'pNN50', 'LF/HF', 'Quality', 'Age', 'Gender', 'Fitness', 'Artifacts'],
                self.risk_classifier.feature_importances_
            ))
        }
    
    def assess_risk(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess cardiovascular risk using trained AI models
        """
        if not self.is_trained:
            self.train_models()
        
        # Extract features
        features = self._extract_features_for_ai(metrics)
        
        if features is None:
            return self._fallback_risk_assessment(metrics)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # AI risk prediction
        risk_proba = self.risk_classifier.predict_proba(features_scaled)[0]
        risk_class = self.risk_classifier.predict(features_scaled)[0]
        trend_score = self.trend_detector.predict(features_scaled)[0]
        
        # Rule-based pattern detection
        pattern_results = self._detect_literature_patterns(metrics)
        
        # Combine AI and rule-based assessments
        risk_labels = ['Low', 'Moderate', 'High']
        ai_risk_level = risk_labels[risk_class]
        
        # Final risk determination (use higher of AI and pattern-based)
        pattern_risk_level = max(pattern_results, key=lambda x: ['Low', 'Moderate', 'High'].index(x['risk_level']) if x else 0)
        if pattern_results:
            pattern_max_risk = max(pattern_results, key=lambda x: ['Low', 'Moderate', 'High'].index(x['risk_level']))['risk_level']
            final_risk_level = pattern_max_risk if ['Low', 'Moderate', 'High'].index(pattern_max_risk) > ['Low', 'Moderate', 'High'].index(ai_risk_level) else ai_risk_level
        else:
            final_risk_level = ai_risk_level
        
        return {
            'ai_risk_assessment': {
                'risk_level': final_risk_level,
                'risk_probabilities': {
                    'low': round(risk_proba[0], 3),
                    'moderate': round(risk_proba[1], 3),
                    'high': round(risk_proba[2], 3)
                },
                'trend_score': round(trend_score, 3),
                'confidence': max(risk_proba)
            },
            'pattern_detection': pattern_results,
            'literature_flags': self._generate_literature_flags(metrics),
            'ai_recommendations': self._generate_ai_recommendations(final_risk_level, trend_score, pattern_results)
        }
    
    def _extract_features_for_ai(self, metrics: Dict[str, float]) -> Optional[List[float]]:
        """Extract and validate features for AI model input"""
        try:
            # Required metrics
            hr = metrics.get('hr', 0)
            rmssd = metrics.get('rmssd', 0)
            sdnn = metrics.get('sdnn', 0)
            pnn50 = metrics.get('pnn50', 0)
            lf_hf = metrics.get('lf_hf_ratio', 0)
            signal_quality = metrics.get('signal_quality', 0)
            
            # Estimated demographic features (simplified)
            age_estimated = 45  # Default middle age
            gender_estimated = 0.5  # Unknown
            fitness_estimated = 0.5  # Average
            artifact_pct = metrics.get('artifact_percentage', 0)
            
            # Validate essential metrics
            if hr == 0 or (rmssd == 0 and sdnn == 0):
                return None
            
            return [hr, rmssd, sdnn, pnn50, lf_hf, signal_quality,
                   age_estimated, gender_estimated, fitness_estimated, artifact_pct]
        
        except Exception:
            return None
    
    def _detect_literature_patterns(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect risk patterns based on cardiovascular literature"""
        
        detected_patterns = []
        
        hr = metrics.get('hr', 70)
        rmssd = metrics.get('rmssd', 30)
        sdnn = metrics.get('sdnn', 40)
        lf_hf = metrics.get('lf_hf_ratio', 1.0)
        
        # Pattern 1: Sudden Cardiac Death Risk (Kleiger et al., 1987)
        if sdnn < 25 and rmssd < 15:
            detected_patterns.append({
                'pattern': 'SuddenCardiacDeathRisk',
                'risk_level': 'High',
                'description': 'Very low HRV indicates increased sudden cardiac death risk',
                'reference': 'Kleiger et al., 1987; Circulation',
                'metrics': {'SDNN': sdnn, 'RMSSD': rmssd}
            })
        
        # Pattern 2: Metabolic Syndrome Risk (Thayer et al., 2010)
        if hr > 80 and rmssd < 25:
            detected_patterns.append({
                'pattern': 'MetabolicSyndromeRisk',
                'risk_level': 'Moderate',
                'description': 'Elevated HR with low HRV suggests metabolic dysfunction',
                'reference': 'Thayer et al., 2010; Neuroscience & Biobehavioral Reviews',
                'metrics': {'HR': hr, 'RMSSD': rmssd}
            })
        
        # Pattern 3: Severe Autonomic Imbalance (Pumprla et al., 2002)
        if lf_hf > 4.0 and rmssd < 20:
            detected_patterns.append({
                'pattern': 'AutonomicDysfunction',
                'risk_level': 'High',
                'description': 'Sympathetic dominance with low parasympathetic activity',
                'reference': 'Pumprla et al., 2002; Clinical Physiology',
                'metrics': {'LF/HF': lf_hf, 'RMSSD': rmssd}
            })
        
        # Pattern 4: Hypertension Risk Pattern
        if hr > 85 and lf_hf > 2.5:
            detected_patterns.append({
                'pattern': 'HypertensionRisk',
                'risk_level': 'Moderate', 
                'description': 'Elevated HR and sympathetic activity suggest hypertension risk',
                'reference': 'Schroeder et al., 2003; American Journal of Hypertension',
                'metrics': {'HR': hr, 'LF/HF': lf_hf}
            })
        
        return detected_patterns
    
    def _generate_literature_flags(self, metrics: Dict[str, float]) -> List[str]:
        """Generate flags based on literature-derived thresholds"""
        
        flags = []
        
        hr = metrics.get('hr', 70)
        rmssd = metrics.get('rmssd', 30)
        sdnn = metrics.get('sdnn', 40)
        lf_hf = metrics.get('lf_hf_ratio', 1.0)
        
        # Literature-based thresholds
        if sdnn < 20:
            flags.append("Critical: SDNN < 20ms (Dekker et al., 2000 - mortality risk)")
        elif sdnn < 30:
            flags.append("Warning: SDNN < 30ms (reduced cardiovascular health)")
        
        if rmssd < 15:
            flags.append("Critical: RMSSD < 15ms (severe autonomic dysfunction)")
        elif rmssd < 20:
            flags.append("Warning: RMSSD < 20ms (reduced vagal activity)")
        
        if lf_hf > 6.0:
            flags.append("Alert: LF/HF > 6.0 (extreme sympathetic dominance)")
        elif lf_hf > 4.0:
            flags.append("Warning: LF/HF > 4.0 (sympathetic overactivity)")
        
        if hr > 120:
            flags.append("Critical: HR > 120 BPM (tachycardia)")
        elif hr < 50:
            flags.append("Critical: HR < 50 BPM (bradycardia)")
        
        return flags
    
    def _generate_ai_recommendations(self, risk_level: str, trend_score: float, 
                                   patterns: List[Dict]) -> List[str]:
        """Generate AI-based clinical recommendations"""
        
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append("Urgent: Consider immediate medical consultation")
            recommendations.append("Monitor symptoms: chest pain, shortness of breath, dizziness")
            
        elif risk_level == 'Moderate':
            recommendations.append("Advisory: Schedule routine cardiovascular check-up")
            recommendations.append("Lifestyle: Focus on stress reduction and regular exercise")
        
        # Pattern-specific recommendations
        for pattern in patterns:
            if pattern['pattern'] == 'MetabolicSyndromeRisk':
                recommendations.append("Metabolic health: Consider diet modification and weight management")
            elif pattern['pattern'] == 'AutonomicDysfunction':
                recommendations.append("Autonomic balance: Practice meditation and breathing exercises")
            elif pattern['pattern'] == 'HypertensionRisk':
                recommendations.append("BP monitoring: Check blood pressure regularly")
        
        # Trend-based recommendations
        if trend_score > 0.7:
            recommendations.append("Trend alert: Multiple risk factors present - comprehensive evaluation needed")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _fallback_risk_assessment(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Fallback assessment when AI models unavailable"""
        
        hr = metrics.get('hr', 70)
        rmssd = metrics.get('rmssd', 30)
        sdnn = metrics.get('sdnn', 40)
        
        # Simple rule-based assessment
        risk_factors = 0
        if hr < 50 or hr > 120: risk_factors += 2
        if rmssd < 20: risk_factors += 1
        if sdnn < 30: risk_factors += 1
        
        if risk_factors >= 3:
            risk_level = 'High'
        elif risk_factors >= 1:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        return {
            'ai_risk_assessment': {
                'risk_level': risk_level,
                'confidence': 0.7,
                'note': 'Fallback assessment - AI models not available'
            },
            'pattern_detection': [],
            'literature_flags': self._generate_literature_flags(metrics),
            'ai_recommendations': ['Basic assessment completed - consider advanced analysis']
        }
    
    def save_models(self, directory: str = "models"):
        """Save trained AI models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if self.is_trained:
            joblib.dump(self.risk_classifier, os.path.join(directory, 'ppg_risk_classifier.pkl'))
            joblib.dump(self.trend_detector, os.path.join(directory, 'ppg_trend_detector.pkl'))
            joblib.dump(self.scaler, os.path.join(directory, 'ppg_feature_scaler.pkl'))
            
            print(f"AI models saved to {directory}/")
    
    def load_models(self, directory: str = "models") -> bool:
        """Load pre-trained AI models"""
        try:
            self.risk_classifier = joblib.load(os.path.join(directory, 'ppg_risk_classifier.pkl'))
            self.trend_detector = joblib.load(os.path.join(directory, 'ppg_trend_detector.pkl'))
            self.scaler = joblib.load(os.path.join(directory, 'ppg_feature_scaler.pkl'))
            self.is_trained = True
            print(f"AI models loaded from {directory}/")
            return True
        except Exception as e:
            print(f"Failed to load AI models: {e}")
            return False

# Global instance for easy access
ppg_ai_assessor = PPGRiskAssessmentAI()

def assess_cardiovascular_risk_ai(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Convenience function for AI-based cardiovascular risk assessment
    
    Args:
        metrics: Dictionary containing PPG-derived cardiovascular metrics
        
    Returns:
        Comprehensive AI risk assessment with patterns and recommendations
    """
    return ppg_ai_assessor.assess_risk(metrics)