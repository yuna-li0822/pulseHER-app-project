"""
PulseHER Longitudinal Data Tracking System
=========================================

Comprehensive system for storing and tracking user metrics over time.
Designed to work transparently with existing app flow without changing visuals.

Features:
- Multi-user data separation
- Time-series storage for all metrics
- Menstrual cycle phase correlation
- Baseline calculation and trending
- Data persistence (JSON + future database integration)
- Privacy-preserving data aggregation

Author: PulseHER Team
Version: 1.0
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import threading

# Thread lock for safe concurrent access
data_lock = threading.Lock()

@dataclass
class MetricDataPoint:
    """Single metric measurement with full context"""
    timestamp: str
    value: float
    metric_name: str
    source: str  # 'ppg', 'manual', 'calculated'
    session_id: Optional[str] = None
    cycle_phase: Optional[str] = None  # 'menstrual', 'follicular', 'ovulation', 'luteal'
    cycle_day: Optional[int] = None
    quality_score: Optional[float] = None
    context_tags: Optional[List[str]] = None  # ['stressed', 'caffeinated', 'exercise']

@dataclass
class PPGSession:
    """Complete PPG measurement session"""
    session_id: str
    user_id: str
    start_timestamp: str
    end_timestamp: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Raw data references
    raw_ppg_data: Optional[List[float]] = None
    rr_intervals: Optional[List[float]] = None
    
    # Computed metrics
    heart_rate_metrics: Optional[Dict] = None
    hrv_metrics: Optional[Dict] = None
    clinical_indices: Optional[Dict] = None
    
    # Quality and context
    signal_quality: Optional[str] = None
    artifact_percentage: Optional[float] = None
    user_notes: Optional[str] = None
    cycle_info: Optional[Dict] = None

@dataclass
class UserProfile:
    """User profile with cycle and health information"""
    user_id: str
    created_at: str
    
    # Cycle information
    cycle_length_days: int = 28
    last_period_start: Optional[str] = None
    period_length_days: int = 5
    
    # Demographics
    age: Optional[int] = None
    birth_assigned_sex: Optional[str] = None
    
    # Health context
    medications: Optional[List[str]] = None
    health_conditions: Optional[List[str]] = None
    lifestyle_factors: Optional[Dict] = None
    
    # Privacy settings
    data_sharing_consent: bool = False
    research_participation: bool = False

class MultiUserLongitudinalManager:
    """
    Comprehensive longitudinal data management system
    Handles multiple users with complete data isolation
    """
    
    def __init__(self, data_directory: str = "user_data"):
        """Initialize the longitudinal manager"""
        self.data_directory = data_directory
        self.ensure_data_directory()
        
        # In-memory caches for performance
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_metrics: Dict[str, List[MetricDataPoint]] = defaultdict(list)
        self.user_sessions: Dict[str, List[PPGSession]] = defaultdict(list)
        
        # Load existing data
        self.load_all_user_data()
        
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(self.data_directory, exist_ok=True)
        
    def get_user_data_path(self, user_id: str) -> str:
        """Get file path for user's data"""
        return os.path.join(self.data_directory, f"user_{user_id}.json")
    
    def load_all_user_data(self):
        """Load all user data from disk"""
        try:
            if not os.path.exists(self.data_directory):
                return
                
            for filename in os.listdir(self.data_directory):
                if filename.startswith("user_") and filename.endswith(".json"):
                    user_id = filename.replace("user_", "").replace(".json", "")
                    self.load_user_data(user_id)
                    
        except Exception as e:
            print(f"[WARN] Error loading user data: {e}")
    
    def load_user_data(self, user_id: str):
        """Load specific user's data from disk"""
        try:
            file_path = self.get_user_data_path(user_id)
            
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Load profile
            if 'profile' in data:
                self.user_profiles[user_id] = UserProfile(**data['profile'])
            
            # Load metrics
            if 'metrics' in data:
                self.user_metrics[user_id] = [
                    MetricDataPoint(**metric) for metric in data['metrics']
                ]
            
            # Load sessions
            if 'sessions' in data:
                self.user_sessions[user_id] = [
                    PPGSession(**session) for session in data['sessions']
                ]
                
        except Exception as e:
            print(f"[WARN] Error loading data for user {user_id}: {e}")
    
    def save_user_data(self, user_id: str):
        """Save specific user's data to disk"""
        try:
            with data_lock:
                file_path = self.get_user_data_path(user_id)
                
                data = {
                    'profile': asdict(self.user_profiles.get(user_id)) if user_id in self.user_profiles else None,
                    'metrics': [asdict(metric) for metric in self.user_metrics[user_id]],
                    'sessions': [asdict(session) for session in self.user_sessions[user_id]],
                    'last_updated': datetime.now().isoformat()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            print(f"[ERROR] Error saving data for user {user_id}: {e}")
    
    # === PROFILE MANAGEMENT ===
    
    def create_or_update_profile(self, user_id: str, profile_data: Dict) -> UserProfile:
        """Create or update user profile"""
        try:
            if user_id in self.user_profiles:
                # Update existing profile
                profile = self.user_profiles[user_id]
                for key, value in profile_data.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
            else:
                # Create new profile
                profile = UserProfile(
                    user_id=user_id,
                    created_at=datetime.now().isoformat(),
                    **profile_data
                )
                self.user_profiles[user_id] = profile
            
            self.save_user_data(user_id)
            return profile
            
        except Exception as e:
            print(f"[ERROR] Profile update failed for {user_id}: {e}")
            raise
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.user_profiles.get(user_id)
    
    # === METRIC TRACKING ===
    
    def add_metric(self, user_id: str, metric_name: str, value: float, 
                  source: str = 'manual', **kwargs) -> MetricDataPoint:
        """Add a single metric measurement"""
        try:
            # Calculate cycle information if profile exists
            cycle_phase, cycle_day = self.calculate_cycle_info(user_id)
            
            metric_point = MetricDataPoint(
                timestamp=datetime.now().isoformat(),
                value=value,
                metric_name=metric_name,
                source=source,
                cycle_phase=cycle_phase,
                cycle_day=cycle_day,
                **kwargs
            )
            
            self.user_metrics[user_id].append(metric_point)
            self.save_user_data(user_id)
            
            return metric_point
            
        except Exception as e:
            print(f"[ERROR] Failed to add metric for {user_id}: {e}")
            raise
    
    def add_metrics_batch(self, user_id: str, metrics_dict: Dict[str, float], 
                         source: str = 'calculated', **kwargs) -> List[MetricDataPoint]:
        """Add multiple metrics from a single measurement session"""
        try:
            added_metrics = []
            
            for metric_name, value in metrics_dict.items():
                if value is not None:  # Skip None values
                    metric_point = self.add_metric(
                        user_id, metric_name, value, source, **kwargs
                    )
                    added_metrics.append(metric_point)
            
            return added_metrics
            
        except Exception as e:
            print(f"[ERROR] Failed to add metrics batch for {user_id}: {e}")
            raise
    
    # === PPG SESSION MANAGEMENT ===
    
    def start_ppg_session(self, user_id: str, session_id: str) -> PPGSession:
        """Start a new PPG measurement session"""
        try:
            cycle_phase, cycle_day = self.calculate_cycle_info(user_id)
            
            session = PPGSession(
                session_id=session_id,
                user_id=user_id,
                start_timestamp=datetime.now().isoformat(),
                cycle_info={
                    'phase': cycle_phase,
                    'cycle_day': cycle_day
                }
            )
            
            self.user_sessions[user_id].append(session)
            self.save_user_data(user_id)
            
            return session
            
        except Exception as e:
            print(f"[ERROR] Failed to start PPG session for {user_id}: {e}")
            raise
    
    def complete_ppg_session(self, user_id: str, session_id: str, 
                           session_data: Dict) -> PPGSession:
        """Complete a PPG session with final data"""
        try:
            # Find the session
            session = None
            for s in self.user_sessions[user_id]:
                if s.session_id == session_id:
                    session = s
                    break
            
            if not session:
                raise ValueError(f"Session {session_id} not found for user {user_id}")
            
            # Update session with completion data
            session.end_timestamp = datetime.now().isoformat()
            session.duration_seconds = session_data.get('duration_seconds')
            session.raw_ppg_data = session_data.get('raw_ppg_data')
            session.rr_intervals = session_data.get('rr_intervals')
            session.heart_rate_metrics = session_data.get('heart_rate_metrics')
            session.hrv_metrics = session_data.get('hrv_metrics')
            session.clinical_indices = session_data.get('clinical_indices')
            session.signal_quality = session_data.get('signal_quality')
            session.artifact_percentage = session_data.get('artifact_percentage')
            session.user_notes = session_data.get('user_notes')
            
            # Add individual metrics to the longitudinal tracking
            if session.heart_rate_metrics:
                self.add_metrics_batch(
                    user_id, 
                    session.heart_rate_metrics, 
                    source='ppg',
                    session_id=session_id,
                    quality_score=session_data.get('quality_score')
                )
            
            if session.hrv_metrics:
                self.add_metrics_batch(
                    user_id, 
                    session.hrv_metrics, 
                    source='ppg',
                    session_id=session_id,
                    quality_score=session_data.get('quality_score')
                )
            
            if session.clinical_indices:
                self.add_metrics_batch(
                    user_id, 
                    session.clinical_indices, 
                    source='ppg',
                    session_id=session_id,
                    quality_score=session_data.get('quality_score')
                )
            
            self.save_user_data(user_id)
            return session
            
        except Exception as e:
            print(f"[ERROR] Failed to complete PPG session for {user_id}: {e}")
            raise
    
    # === CYCLE CALCULATIONS ===
    
    def calculate_cycle_info(self, user_id: str) -> Tuple[Optional[str], Optional[int]]:
        """Calculate current menstrual cycle phase and day"""
        try:
            profile = self.user_profiles.get(user_id)
            
            if not profile or not profile.last_period_start:
                return None, None
            
            # Parse last period date
            last_period = datetime.fromisoformat(profile.last_period_start.replace('Z', '+00:00'))
            now = datetime.now()
            
            # Calculate days since last period
            days_since_period = (now - last_period).days
            
            # Calculate cycle day (1-based)
            cycle_day = (days_since_period % profile.cycle_length_days) + 1
            
            # Determine phase based on cycle day
            if cycle_day <= profile.period_length_days:
                phase = 'menstrual'
            elif cycle_day <= profile.cycle_length_days // 2 - 2:
                phase = 'follicular'
            elif cycle_day <= profile.cycle_length_days // 2 + 2:
                phase = 'ovulation'
            else:
                phase = 'luteal'
            
            return phase, cycle_day
            
        except Exception as e:
            print(f"[WARN] Cycle calculation failed for {user_id}: {e}")
            return None, None
    
    # === DATA RETRIEVAL ===
    
    def get_user_metrics(self, user_id: str, metric_names: Optional[List[str]] = None,
                        days_back: int = 30) -> List[MetricDataPoint]:
        """Get user's metrics for specified time period"""
        try:
            since_date = datetime.now() - timedelta(days=days_back)
            
            metrics = []
            for metric in self.user_metrics[user_id]:
                metric_time = datetime.fromisoformat(metric.timestamp.replace('Z', '+00:00'))
                
                # Filter by date
                if metric_time >= since_date:
                    # Filter by metric names if specified
                    if metric_names is None or metric.metric_name in metric_names:
                        metrics.append(metric)
            
            # Sort by timestamp
            metrics.sort(key=lambda m: m.timestamp)
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Failed to get metrics for {user_id}: {e}")
            return []
    
    def get_user_sessions(self, user_id: str, days_back: int = 30) -> List[PPGSession]:
        """Get user's PPG sessions for specified time period"""
        try:
            since_date = datetime.now() - timedelta(days=days_back)
            
            sessions = []
            for session in self.user_sessions[user_id]:
                session_time = datetime.fromisoformat(session.start_timestamp.replace('Z', '+00:00'))
                
                if session_time >= since_date:
                    sessions.append(session)
            
            # Sort by timestamp
            sessions.sort(key=lambda s: s.start_timestamp)
            return sessions
            
        except Exception as e:
            print(f"[ERROR] Failed to get sessions for {user_id}: {e}")
            return []
    
    # === ANALYTICS AND INSIGHTS ===
    
    def calculate_user_baselines(self, user_id: str, metric_name: str, 
                               phase: Optional[str] = None) -> Dict[str, float]:
        """Calculate baseline statistics for a specific metric"""
        try:
            # Get relevant metrics
            metrics = self.get_user_metrics(user_id, [metric_name], days_back=90)
            
            # Filter by cycle phase if specified
            if phase:
                metrics = [m for m in metrics if m.cycle_phase == phase]
            
            if len(metrics) < 2:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'sample_count': len(values),
                'trend': self.calculate_trend(values)
            }
            
        except Exception as e:
            print(f"[ERROR] Baseline calculation failed for {user_id}: {e}")
            return {}
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend from values (improving/declining/stable)"""
        try:
            if len(values) < 3:
                return 'insufficient_data'
            
            # Simple linear trend calculation
            recent_avg = statistics.mean(values[-len(values)//3:])
            older_avg = statistics.mean(values[:len(values)//3])
            
            change_pct = ((recent_avg - older_avg) / older_avg) * 100
            
            if abs(change_pct) < 5:
                return 'stable'
            elif change_pct > 0:
                return 'improving'
            else:
                return 'declining'
                
        except Exception:
            return 'unknown'
    
    def get_dashboard_summary(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive dashboard data for user"""
        try:
            metrics = self.get_user_metrics(user_id, days_back=days_back)
            sessions = self.get_user_sessions(user_id, days_back=days_back)
            profile = self.get_profile(user_id)
            
            # Group metrics by name
            metrics_by_name = defaultdict(list)
            for metric in metrics:
                metrics_by_name[metric.metric_name].append(metric)
            
            # Calculate summaries
            metric_summaries = {}
            for metric_name, metric_list in metrics_by_name.items():
                values = [m.value for m in metric_list]
                if values:
                    metric_summaries[metric_name] = {
                        'latest_value': values[-1],
                        'latest_timestamp': metric_list[-1].timestamp,
                        'mean': statistics.mean(values),
                        'count': len(values),
                        'trend': self.calculate_trend(values)
                    }
            
            return {
                'user_id': user_id,
                'profile': asdict(profile) if profile else None,
                'date_range': {
                    'start': (datetime.now() - timedelta(days=days_back)).isoformat(),
                    'end': datetime.now().isoformat()
                },
                'total_metrics_count': len(metrics),
                'total_sessions_count': len(sessions),
                'metric_summaries': metric_summaries,
                'recent_sessions': [asdict(s) for s in sessions[-5:]],  # Last 5 sessions
                'cycle_info': self.calculate_cycle_info(user_id)
            }
            
        except Exception as e:
            print(f"[ERROR] Dashboard summary failed for {user_id}: {e}")
            return {'error': str(e)}
    
    # === DATA MANAGEMENT ===
    
    def cleanup_old_data(self, user_id: str, days_to_keep: int = 365):
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean metrics
            original_count = len(self.user_metrics[user_id])
            self.user_metrics[user_id] = [
                m for m in self.user_metrics[user_id]
                if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')) >= cutoff_date
            ]
            
            # Clean sessions  
            original_sessions = len(self.user_sessions[user_id])
            self.user_sessions[user_id] = [
                s for s in self.user_sessions[user_id]
                if datetime.fromisoformat(s.start_timestamp.replace('Z', '+00:00')) >= cutoff_date
            ]
            
            self.save_user_data(user_id)
            
            print(f"[INFO] Cleaned {original_count - len(self.user_metrics[user_id])} old metrics")
            print(f"[INFO] Cleaned {original_sessions - len(self.user_sessions[user_id])} old sessions")
            
        except Exception as e:
            print(f"[ERROR] Data cleanup failed for {user_id}: {e}")
    
    def export_user_data(self, user_id: str, anonymize: bool = True) -> Dict:
        """Export user data for research or backup"""
        try:
            data = {
                'profile': asdict(self.user_profiles.get(user_id)) if user_id in self.user_profiles else None,
                'metrics': [asdict(m) for m in self.user_metrics[user_id]],
                'sessions': [asdict(s) for s in self.user_sessions[user_id]],
                'export_timestamp': datetime.now().isoformat()
            }
            
            if anonymize and data['profile']:
                # Remove identifying information
                data['profile']['user_id'] = 'anonymized'
                if 'personal_info' in data['profile']:
                    del data['profile']['personal_info']
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Data export failed for {user_id}: {e}")
            return {}

# Global instance for the app
longitudinal_manager = None

def get_longitudinal_manager() -> MultiUserLongitudinalManager:
    """Get or create the global longitudinal manager instance"""
    global longitudinal_manager
    
    if longitudinal_manager is None:
        longitudinal_manager = MultiUserLongitudinalManager()
    
    return longitudinal_manager

def initialize_longitudinal_tracking():
    """Initialize the longitudinal tracking system"""
    try:
        manager = get_longitudinal_manager()
        print(f"[OK] Longitudinal tracking initialized")
        print(f"[INFO] Data directory: {manager.data_directory}")
        print(f"[INFO] Loaded {len(manager.user_profiles)} user profiles")
        
        total_metrics = sum(len(metrics) for metrics in manager.user_metrics.values())
        total_sessions = sum(len(sessions) for sessions in manager.user_sessions.values())
        
        print(f"[INFO] Total metrics tracked: {total_metrics}")
        print(f"[INFO] Total sessions recorded: {total_sessions}")
        
        return manager
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize longitudinal tracking: {e}")
        return None

# Convenience functions for easy integration

def save_ppg_session_data(user_id: str, session_id: str, ppg_results: Dict):
    """Save PPG session data to longitudinal tracking"""
    try:
        manager = get_longitudinal_manager()
        return manager.complete_ppg_session(user_id, session_id, ppg_results)
    except Exception as e:
        print(f"[WARN] Failed to save PPG session data: {e}")
        return None

def save_user_metrics(user_id: str, metrics: Dict[str, float], source: str = 'manual'):
    """Save user metrics to longitudinal tracking"""
    try:
        manager = get_longitudinal_manager()
        return manager.add_metrics_batch(user_id, metrics, source)
    except Exception as e:
        print(f"[WARN] Failed to save user metrics: {e}")
        return None

def update_user_profile(user_id: str, profile_data: Dict):
    """Update user profile in longitudinal tracking"""
    try:
        manager = get_longitudinal_manager()
        return manager.create_or_update_profile(user_id, profile_data)
    except Exception as e:
        print(f"[WARN] Failed to update user profile: {e}")
        return None

def get_user_dashboard_data(user_id: str, days: int = 30):
    """Get dashboard data for user from longitudinal tracking"""
    try:
        manager = get_longitudinal_manager()
        return manager.get_dashboard_summary(user_id, days)
    except Exception as e:
        print(f"[WARN] Failed to get dashboard data: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test the system
    print("Testing Longitudinal Tracking System...")
    
    manager = initialize_longitudinal_tracking()
    
    if manager:
        # Test user creation
        test_user = "test_user_123"
        
        # Create profile
        profile = manager.create_or_update_profile(test_user, {
            'cycle_length_days': 28,
            'last_period_start': '2025-10-01',
            'age': 25,
            'birth_assigned_sex': 'female'
        })
        print(f"Created profile: {profile.user_id}")
        
        # Add some test metrics
        metrics = manager.add_metrics_batch(test_user, {
            'heart_rate': 72.5,
            'rmssd': 35.2,
            'sdnn': 45.1,
            'stress_score': 25.0
        }, source='test')
        print(f"Added {len(metrics)} metrics")
        
        # Get dashboard summary
        dashboard = manager.get_dashboard_summary(test_user)
        print(f"Dashboard summary: {dashboard['total_metrics_count']} metrics tracked")
        
        print("✅ Longitudinal tracking system test successful!")
    else:
        print("❌ Longitudinal tracking system test failed!")
