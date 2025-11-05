"""
PulseHER Database Schema - Blueprint Implementation
Complete database models following the comprehensive blueprint
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    """
    Blueprint: users table with comprehensive profile data
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # Demographics
    date_of_birth = Column(DateTime)
    sex_at_birth = Column(String(10))  # 'female', 'male', 'intersex'
    gender_identity = Column(String(50))
    weight_kg = Column(Float)
    height_cm = Column(Float)
    
    # Medical history (JSON for flexibility)
    medications = Column(JSON)  # ["beta_blocker", "contraceptive", "ssri"]
    conditions = Column(JSON)   # ["hypertension", "pcos", "thyroid"]
    
    # Cycle information
    cycle_length_days = Column(Integer, default=28)
    last_period_start_date = Column(DateTime)
    contraceptive_usage = Column(String(100))
    menopause_status = Column(String(50))
    
    # Lifestyle
    typical_sleep_hours = Column(Float)
    activity_level = Column(String(50))  # 'sedentary', 'moderate', 'active'
    
    # Privacy & consent
    consent_research = Column(Boolean, default=False)
    consent_clinical_share = Column(Boolean, default=False)
    privacy_preferences = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    baselines = relationship("Baseline", back_populates="user", cascade="all, delete-orphan")
    flags = relationship("ClinicalFlag", back_populates="user", cascade="all, delete-orphan")

class Session(Base):
    """
    Blueprint: sessions table for PPG recordings
    """
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Session metadata
    timestamp_utc = Column(DateTime, nullable=False, default=datetime.utcnow)
    device_type = Column(String(50))  # 'camera_ppg', 'wearable', 'manual'
    duration_seconds = Column(Float)
    sampling_rate_hz = Column(Float)
    
    # Data storage options
    rr_intervals_json = Column(Text)  # JSON string of RR intervals
    raw_ppg_location = Column(String(500))  # S3/file path if storing raw PPG
    
    # Quality metrics
    artifact_percentage = Column(Float)
    quality_flag = Column(String(20))  # 'excellent', 'good', 'fair', 'poor'
    
    # Context tags
    user_tags = Column(JSON)  # ["stressed", "caffeinated", "post_exercise"]
    notes = Column(Text)
    
    # Processing metadata
    processing_version = Column(String(20))
    processed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    metrics = relationship("SessionMetrics", back_populates="session", uselist=False)
    indices = relationship("ClinicalIndices", back_populates="session", uselist=False)
    flags = relationship("ClinicalFlag", back_populates="session")

class SessionMetrics(Base):
    """
    Blueprint: computed metrics for each session
    """
    __tablename__ = 'session_metrics'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, unique=True)
    
    # Time domain metrics
    mean_hr_bpm = Column(Float)
    sdnn_ms = Column(Float)
    rmssd_ms = Column(Float)
    pnn50_pct = Column(Float)
    mean_rr_ms = Column(Float)
    
    # Frequency domain metrics
    vlf_power = Column(Float)
    lf_power = Column(Float)
    hf_power = Column(Float)
    total_power = Column(Float)
    lf_hf_ratio = Column(Float)
    lf_norm = Column(Float)
    hf_norm = Column(Float)
    
    # Additional derived metrics
    respiration_estimate = Column(Float)
    stress_score = Column(Float)
    
    # Processing metadata
    valid_intervals_count = Column(Integer)
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="metrics")

class ClinicalIndices(Base):
    """
    Blueprint: Advanced clinical indices (ABI, CVR, CSI)
    """
    __tablename__ = 'clinical_indices'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, unique=True)
    
    # Core indices
    abi = Column(Float)  # Autonomic Balance Index (0-100)
    cvr = Column(Float)  # Cardiovascular Risk (0-100)
    csi = Column(Float)  # Cardiac Stress Index (0-100)
    
    # Z-scores relative to personal baseline
    z_scores = Column(JSON)
    
    # Explanations and confidence
    explanation_json = Column(JSON)
    confidence_scores = Column(JSON)
    
    # Cycle awareness
    cycle_phase = Column(String(20))  # 'menstrual', 'follicular', 'ovulation', 'luteal'
    cycle_day = Column(Integer)
    phase_adjusted = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("Session", back_populates="indices")

class Baseline(Base):
    """
    Blueprint: Per-user per-phase baselines for normalization
    """
    __tablename__ = 'baselines'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Baseline parameters
    phase = Column(String(20))  # 'menstrual', 'follicular', 'ovulation', 'luteal', 'overall'
    metric_name = Column(String(50))  # 'mean_hr_bpm', 'rmssd_ms', etc.
    
    # Statistical parameters
    baseline_mean = Column(Float)
    baseline_std = Column(Float)
    sample_count = Column(Integer)
    
    # Update tracking
    last_updated = Column(DateTime, default=datetime.utcnow)
    update_count = Column(Integer, default=1)
    
    # Relationships
    user = relationship("User", back_populates="baselines")

class ClinicalFlag(Base):
    """
    Blueprint: Clinical flags and alerts
    """
    __tablename__ = 'clinical_flags'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    
    # Flag details
    flag_type = Column(String(50))  # 'tachycardia', 'low_hrv', 'cardiovascular_risk'
    severity = Column(String(20))   # 'info', 'warning', 'alert', 'critical'
    confidence = Column(Float)      # 0.0 - 1.0
    message = Column(Text)
    
    # Clinical context
    clinical_category = Column(String(50))  # 'cardiac', 'autonomic', 'quality'
    actionable = Column(Boolean, default=True)
    
    # Resolution tracking
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    clinician_reviewed = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="flags")
    session = relationship("Session", back_populates="flags")

class ResearchExport(Base):
    """
    Blueprint: Anonymized research data exports
    """
    __tablename__ = 'research_exports'
    
    id = Column(Integer, primary_key=True)
    
    # Anonymized user demographics
    age_range = Column(String(20))  # '20-25', '26-30', etc.
    sex_at_birth = Column(String(10))
    weight_category = Column(String(20))  # 'underweight', 'normal', 'overweight'
    
    # Anonymized session data
    session_metrics = Column(JSON)
    clinical_indices = Column(JSON)
    cycle_phase = Column(String(20))
    
    # Research metadata
    export_version = Column(String(20))
    research_cohort = Column(String(50))
    consent_level = Column(String(50))
    
    # De-identification
    original_user_hash = Column(String(64))  # SHA256 of user_id + salt
    export_date = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """
    Blueprint: Database operations manager
    """
    
    def __init__(self, database_url="sqlite:///pulseher.db"):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_session_complete(self, user_id, session_data, metrics, indices, flags):
        """
        Save complete session with all computed data
        Blueprint: Complete session storage
        """
        db = self.get_session()
        try:
            # Create session record
            session = Session(
                user_id=user_id,
                timestamp_utc=datetime.utcnow(),
                device_type=session_data.get('device_type', 'camera_ppg'),
                duration_seconds=session_data.get('duration', 30),
                sampling_rate_hz=session_data.get('sampling_rate', 30),
                rr_intervals_json=json.dumps(session_data.get('rr_intervals', [])),
                artifact_percentage=session_data.get('artifact_pct', 0),
                quality_flag=session_data.get('quality_flag', 'unknown'),
                user_tags=session_data.get('tags', []),
                processing_version="4.0",
                processed_at=datetime.utcnow()
            )
            db.add(session)
            db.flush()  # Get session ID
            
            # Save metrics
            if metrics:
                session_metrics = SessionMetrics(
                    session_id=session.id,
                    **metrics
                )
                db.add(session_metrics)
            
            # Save clinical indices
            if indices:
                clinical_indices = ClinicalIndices(
                    session_id=session.id,
                    **indices
                )
                db.add(clinical_indices)
            
            # Save flags
            for flag_data in flags:
                flag = ClinicalFlag(
                    user_id=user_id,
                    session_id=session.id,
                    **flag_data,
                    created_at=datetime.utcnow()
                )
                db.add(flag)
            
            db.commit()
            return session.id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database save error: {e}")
            raise
        finally:
            db.close()
    
    def update_user_baselines(self, user_id, phase, new_metrics):
        """
        Update user baselines with new session data
        Blueprint: Baseline management
        """
        db = self.get_session()
        try:
            for metric_name, value in new_metrics.items():
                if value is None:
                    continue
                    
                # Get existing baseline
                baseline = db.query(Baseline).filter_by(
                    user_id=user_id,
                    phase=phase,
                    metric_name=metric_name
                ).first()
                
                if baseline:
                    # Update existing baseline (running average)
                    total = baseline.baseline_mean * baseline.sample_count + value
                    baseline.sample_count += 1
                    baseline.baseline_mean = total / baseline.sample_count
                    
                    # Update standard deviation (simplified)
                    # In production, use proper online algorithm
                    baseline.last_updated = datetime.utcnow()
                    baseline.update_count += 1
                else:
                    # Create new baseline
                    baseline = Baseline(
                        user_id=user_id,
                        phase=phase,
                        metric_name=metric_name,
                        baseline_mean=value,
                        baseline_std=0,  # Will be calculated after more data
                        sample_count=1
                    )
                    db.add(baseline)
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Baseline update error: {e}")
            raise
        finally:
            db.close()
    
    def get_user_dashboard_data(self, user_id, days=30):
        """
        Get dashboard data for user
        Blueprint: Dashboard data aggregation
        """
        db = self.get_session()
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Get recent sessions with metrics
            sessions = db.query(Session).filter(
                Session.user_id == user_id,
                Session.timestamp_utc >= since_date
            ).order_by(Session.timestamp_utc.desc()).all()
            
            dashboard_data = {
                'session_count': len(sessions),
                'date_range': {'start': since_date.isoformat(), 'end': datetime.utcnow().isoformat()},
                'sessions': [],
                'trends': {},
                'recent_flags': []
            }
            
            # Aggregate session data
            for session in sessions:
                session_summary = {
                    'id': session.id,
                    'timestamp': session.timestamp_utc.isoformat(),
                    'quality': session.quality_flag,
                    'metrics': {},
                    'indices': {},
                    'cycle_phase': None
                }
                
                if session.metrics:
                    session_summary['metrics'] = {
                        'hr': session.metrics.mean_hr_bpm,
                        'rmssd': session.metrics.rmssd_ms,
                        'sdnn': session.metrics.sdnn_ms,
                        'lf_hf': session.metrics.lf_hf_ratio
                    }
                
                if session.indices:
                    session_summary['indices'] = {
                        'abi': session.indices.abi,
                        'cvr': session.indices.cvr,
                        'csi': session.indices.csi
                    }
                    session_summary['cycle_phase'] = session.indices.cycle_phase
                
                dashboard_data['sessions'].append(session_summary)
            
            # Get recent flags
            recent_flags = db.query(ClinicalFlag).filter(
                ClinicalFlag.user_id == user_id,
                ClinicalFlag.created_at >= since_date,
                ClinicalFlag.resolved == False
            ).order_by(ClinicalFlag.created_at.desc()).limit(10).all()
            
            dashboard_data['recent_flags'] = [
                {
                    'type': flag.flag_type,
                    'severity': flag.severity,
                    'message': flag.message,
                    'timestamp': flag.created_at.isoformat()
                }
                for flag in recent_flags
            ]
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard data error: {e}")
            return None
        finally:
            db.close()

# Global database manager instance
db_manager = DatabaseManager()

def init_database(database_url=None):
    """Initialize database with custom URL if provided"""
    global db_manager
    if database_url:
        db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager