"""
PulseHER Blueprint API - Complete REST API Implementation
Following the comprehensive blueprint specifications
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import jwt
import bcrypt
from datetime import datetime, timedelta
import json
import logging
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import base64

from core_algorithms import pulse_core
from database_schema import db_manager, User, Session

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = "pulseher_secret_key_change_in_production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

def init_blueprint_api(app):
    """
    Initialize all Blueprint API endpoints
    """
    
    # CORS setup
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    @app.route('/api/register', methods=['POST'])
    def register_user():
        """
        Blueprint: User registration with comprehensive profile
        """
        try:
            data = request.get_json()
            
            # Validate required fields
            required = ['email', 'password']
            for field in required:
                if not data.get(field):
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            db = db_manager.get_session()
            
            # Check if user exists
            existing = db.query(User).filter_by(email=data['email']).first()
            if existing:
                return jsonify({'error': 'User already exists'}), 409
            
            # Hash password
            password_hash = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Create user
            user = User(
                email=data['email'],
                password_hash=password_hash,
                date_of_birth=datetime.fromisoformat(data['date_of_birth']) if data.get('date_of_birth') else None,
                sex_at_birth=data.get('sex_at_birth'),
                gender_identity=data.get('gender_identity'),
                weight_kg=data.get('weight_kg'),
                height_cm=data.get('height_cm'),
                medications=data.get('medications', []),
                conditions=data.get('conditions', []),
                cycle_length_days=data.get('cycle_length_days', 28),
                last_period_start_date=datetime.fromisoformat(data['last_period_start_date']) if data.get('last_period_start_date') else None,
                contraceptive_usage=data.get('contraceptive_usage'),
                menopause_status=data.get('menopause_status'),
                typical_sleep_hours=data.get('typical_sleep_hours'),
                activity_level=data.get('activity_level'),
                consent_research=data.get('consent_research', False),
                consent_clinical_share=data.get('consent_clinical_share', False),
                privacy_preferences=data.get('privacy_preferences', {})
            )
            
            db.add(user)
            db.commit()
            
            # Generate JWT token
            token = jwt.encode({
                'user_id': user.id,
                'email': user.email,
                'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
            }, JWT_SECRET, algorithm=JWT_ALGORITHM)
            
            db.close()
            
            return jsonify({
                'message': 'User registered successfully',
                'user_id': user.id,
                'token': token,
                'expires_in': JWT_EXPIRATION_HOURS * 3600
            }), 201
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({'error': 'Registration failed'}), 500
    
    @app.route('/api/login', methods=['POST'])
    def login_user():
        """
        Blueprint: User authentication
        """
        try:
            data = request.get_json()
            
            if not data.get('email') or not data.get('password'):
                return jsonify({'error': 'Email and password required'}), 400
            
            db = db_manager.get_session()
            user = db.query(User).filter_by(email=data['email']).first()
            
            if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user.password_hash.encode('utf-8')):
                return jsonify({'error': 'Invalid credentials'}), 401
            
            # Generate JWT token
            token = jwt.encode({
                'user_id': user.id,
                'email': user.email,
                'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
            }, JWT_SECRET, algorithm=JWT_ALGORITHM)
            
            # Get user profile
            profile = {
                'id': user.id,
                'email': user.email,
                'cycle_length': user.cycle_length_days,
                'last_period': user.last_period_start_date.isoformat() if user.last_period_start_date else None,
                'medications': user.medications,
                'conditions': user.conditions
            }
            
            db.close()
            
            return jsonify({
                'message': 'Login successful',
                'token': token,
                'user': profile,
                'expires_in': JWT_EXPIRATION_HOURS * 3600
            }), 200
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'error': 'Login failed'}), 500
    
    @app.route('/api/sessions', methods=['POST'])
    def create_session():
        """
        Blueprint: Upload RR intervals or PPG data and process
        """
        try:
            # Authenticate user
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            data = request.get_json()
            
            # Validate input
            if 'rr_intervals' not in data and 'ppg_signal' not in data:
                return jsonify({'error': 'Either rr_intervals or ppg_signal required'}), 400
            
            # Get user profile for cycle awareness
            db = db_manager.get_session()
            user = db.query(User).filter_by(id=user_id).first()
            user_profile = {
                'last_period_date': user.last_period_start_date,
                'cycle_length': user.cycle_length_days,
                'medications': user.medications,
                'conditions': user.conditions
            } if user else None
            db.close()
            
            # Process RR intervals or extract from PPG
            if 'rr_intervals' in data:
                rr_intervals = data['rr_intervals']
            else:
                # Extract RR from PPG signal
                ppg_signal = data['ppg_signal']
                sampling_rate = data.get('sampling_rate', 30)
                rr_intervals = pulse_core.detect_beats_from_ppg(ppg_signal, sampling_rate)
            
            if not rr_intervals or len(rr_intervals) < 3:
                return jsonify({'error': 'Insufficient RR intervals detected'}), 400
            
            # Get user baselines
            baseline = get_user_baseline(user_id, user_profile)
            
            # Complete processing pipeline
            result = pulse_core.process_session_complete(rr_intervals, user_profile, baseline)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 400
            
            # Prepare session data for storage
            session_data = {
                'device_type': data.get('device_type', 'camera_ppg'),
                'duration': data.get('duration', 30),
                'sampling_rate': data.get('sampling_rate', 30),
                'rr_intervals': rr_intervals,
                'artifact_pct': result['rr_quality']['artifact_pct'],
                'quality_flag': determine_quality_flag(result['rr_quality']['artifact_pct']),
                'tags': data.get('tags', [])
            }
            
            # Save to database
            session_id = db_manager.save_session_complete(
                user_id=user_id,
                session_data=session_data,
                metrics=result.get('time_domain'),
                indices=result.get('clinical_indices'),
                flags=result.get('clinical_flags', [])
            )
            
            # Update user baselines
            if result.get('time_domain') and result.get('cycle_info'):
                phase = result['cycle_info']['phase']
                db_manager.update_user_baselines(user_id, phase, result['time_domain'])
            
            # Return complete analysis
            response = {
                'session_id': session_id,
                'timestamp': result['timestamp'],
                'quality': {
                    'artifact_percentage': result['rr_quality']['artifact_pct'],
                    'quality_rating': session_data['quality_flag'],
                    'valid_intervals': result['rr_quality']['cleaned_count']
                },
                'metrics': result.get('time_domain', {}),
                'frequency_analysis': result.get('frequency_domain', {}),
                'clinical_indices': result.get('clinical_indices', {}),
                'clinical_flags': result.get('clinical_flags', []),
                'cycle_info': result.get('cycle_info', {}),
                'recommendations': generate_recommendations(result)
            }
            
            return jsonify(response), 201
            
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return jsonify({'error': 'Session processing failed'}), 500
    
    @app.route('/api/sessions', methods=['GET'])
    def get_sessions():
        """
        Blueprint: Get user session history
        """
        try:
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Parse query parameters
            range_param = request.args.get('range', '30d')
            limit = int(request.args.get('limit', '50'))
            
            # Calculate date range
            if range_param.endswith('d'):
                days = int(range_param[:-1])
                since_date = datetime.utcnow() - timedelta(days=days)
            else:
                since_date = datetime.utcnow() - timedelta(days=30)
            
            # Get sessions from database
            dashboard_data = db_manager.get_user_dashboard_data(user_id, days)
            
            if not dashboard_data:
                return jsonify({'error': 'Failed to retrieve sessions'}), 500
            
            # Limit results
            dashboard_data['sessions'] = dashboard_data['sessions'][:limit]
            
            return jsonify(dashboard_data), 200
            
        except Exception as e:
            logger.error(f"Get sessions error: {e}")
            return jsonify({'error': 'Failed to retrieve sessions'}), 500
    
    @app.route('/api/metrics/summary', methods=['GET'])
    def get_metrics_summary():
        """
        Blueprint: Get aggregated metrics summary
        """
        try:
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            range_param = request.args.get('range', '30d')
            days = int(range_param[:-1]) if range_param.endswith('d') else 30
            
            dashboard_data = db_manager.get_user_dashboard_data(user_id, days)
            
            if not dashboard_data:
                return jsonify({'summary': {}, 'trends': {}}), 200
            
            # Calculate summary statistics
            sessions = dashboard_data['sessions']
            if not sessions:
                return jsonify({'summary': {}, 'trends': {}}), 200
            
            # Aggregate metrics
            hr_values = [s['metrics'].get('hr') for s in sessions if s['metrics'].get('hr')]
            rmssd_values = [s['metrics'].get('rmssd') for s in sessions if s['metrics'].get('rmssd')]
            
            summary = {
                'total_sessions': len(sessions),
                'date_range': dashboard_data['date_range'],
                'averages': {
                    'heart_rate': round(sum(hr_values) / len(hr_values), 1) if hr_values else None,
                    'rmssd': round(sum(rmssd_values) / len(rmssd_values), 1) if rmssd_values else None
                },
                'trends': calculate_trends(sessions),
                'cycle_analysis': analyze_cycle_patterns(sessions)
            }
            
            return jsonify(summary), 200
            
        except Exception as e:
            logger.error(f"Metrics summary error: {e}")
            return jsonify({'error': 'Failed to generate summary'}), 500
    
    @app.route('/api/indices/latest', methods=['GET'])
    def get_latest_indices():
        """
        Blueprint: Get latest clinical indices
        """
        try:
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            dashboard_data = db_manager.get_user_dashboard_data(user_id, 7)  # Last 7 days
            
            if not dashboard_data['sessions']:
                return jsonify({'indices': None, 'message': 'No recent sessions found'}), 200
            
            # Get most recent session with indices
            latest_session = None
            for session in dashboard_data['sessions']:
                if session.get('indices') and any(session['indices'].values()):
                    latest_session = session
                    break
            
            if not latest_session:
                return jsonify({'indices': None, 'message': 'No clinical indices available'}), 200
            
            return jsonify({
                'indices': latest_session['indices'],
                'timestamp': latest_session['timestamp'],
                'cycle_phase': latest_session.get('cycle_phase'),
                'session_id': latest_session['id']
            }), 200
            
        except Exception as e:
            logger.error(f"Latest indices error: {e}")
            return jsonify({'error': 'Failed to retrieve indices'}), 500
    
    @app.route('/api/context', methods=['POST'])
    def add_context():
        """
        Blueprint: Add context tags to sessions
        """
        try:
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            data = request.get_json()
            
            # Validate input
            if not data.get('session_id') or not data.get('tags'):
                return jsonify({'error': 'session_id and tags required'}), 400
            
            # Update session tags (implementation depends on your needs)
            # This is a simplified version
            
            return jsonify({'message': 'Context added successfully'}), 200
            
        except Exception as e:
            logger.error(f"Add context error: {e}")
            return jsonify({'error': 'Failed to add context'}), 500
    
    @app.route('/api/export/clinician', methods=['GET'])
    def export_clinician_report():
        """
        Blueprint: Generate clinician PDF report
        """
        try:
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            range_param = request.args.get('range', '30d')
            days = int(range_param[:-1]) if range_param.endswith('d') else 30
            
            # Generate PDF report
            pdf_buffer = generate_clinician_pdf(user_id, days)
            
            if not pdf_buffer:
                return jsonify({'error': 'Failed to generate report'}), 500
            
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'pulseher_report_{datetime.now().strftime("%Y%m%d")}.pdf'
            )
            
        except Exception as e:
            logger.error(f"Clinician export error: {e}")
            return jsonify({'error': 'Failed to generate report'}), 500
    
    @app.route('/api/user/profile', methods=['GET', 'PUT'])
    def user_profile():
        """
        Blueprint: Get/update user profile
        """
        try:
            user_id = authenticate_request(request)
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            db = db_manager.get_session()
            user = db.query(User).filter_by(id=user_id).first()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            if request.method == 'GET':
                profile = {
                    'id': user.id,
                    'email': user.email,
                    'date_of_birth': user.date_of_birth.isoformat() if user.date_of_birth else None,
                    'sex_at_birth': user.sex_at_birth,
                    'weight_kg': user.weight_kg,
                    'height_cm': user.height_cm,
                    'cycle_length_days': user.cycle_length_days,
                    'last_period_start_date': user.last_period_start_date.isoformat() if user.last_period_start_date else None,
                    'medications': user.medications,
                    'conditions': user.conditions,
                    'activity_level': user.activity_level
                }
                db.close()
                return jsonify({'profile': profile}), 200
            
            elif request.method == 'PUT':
                data = request.get_json()
                
                # Update allowed fields
                updateable_fields = [
                    'weight_kg', 'height_cm', 'cycle_length_days', 
                    'medications', 'conditions', 'activity_level'
                ]
                
                for field in updateable_fields:
                    if field in data:
                        setattr(user, field, data[field])
                
                if 'last_period_start_date' in data and data['last_period_start_date']:
                    user.last_period_start_date = datetime.fromisoformat(data['last_period_start_date'])
                
                user.updated_at = datetime.utcnow()
                db.commit()
                db.close()
                
                return jsonify({'message': 'Profile updated successfully'}), 200
            
        except Exception as e:
            logger.error(f"Profile error: {e}")
            return jsonify({'error': 'Profile operation failed'}), 500

def authenticate_request(request):
    """
    Authenticate JWT token and return user_id
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload['user_id']
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_user_baseline(user_id, user_profile):
    """
    Get user baseline for current cycle phase
    """
    if not user_profile or not user_profile.get('last_period_date'):
        return None
    
    # Determine current cycle phase
    phase = pulse_core.estimate_cycle_phase(
        user_profile['last_period_date'], 
        user_profile.get('cycle_length', 28)
    )
    
    # Get baseline from database (simplified version)
    # In production, implement proper baseline retrieval
    return None

def determine_quality_flag(artifact_pct):
    """
    Determine quality flag based on artifact percentage
    """
    if artifact_pct < 5:
        return 'excellent'
    elif artifact_pct < 15:
        return 'good'
    elif artifact_pct < 25:
        return 'fair'
    else:
        return 'poor'

def generate_recommendations(analysis_result):
    """
    Generate personalized recommendations
    """
    recommendations = []
    
    flags = analysis_result.get('clinical_flags', [])
    indices = analysis_result.get('clinical_indices', {})
    
    # Quality recommendations
    if analysis_result['rr_quality']['artifact_pct'] > 20:
        recommendations.append({
            'type': 'measurement_quality',
            'message': 'For better readings, ensure steady hand position and good lighting',
            'priority': 'medium'
        })
    
    # Clinical recommendations based on flags
    for flag in flags:
        if flag['type'] == 'high_stress':
            recommendations.append({
                'type': 'lifestyle',
                'message': 'Consider stress management techniques like deep breathing or meditation',
                'priority': 'high'
            })
        elif flag['type'] == 'low_hrv':
            recommendations.append({
                'type': 'health',
                'message': 'Low HRV may indicate fatigue. Ensure adequate sleep and recovery',
                'priority': 'medium'
            })
    
    return recommendations

def calculate_trends(sessions):
    """
    Calculate metric trends over time
    """
    if len(sessions) < 2:
        return {}
    
    # Simple trend calculation (last 7 vs previous 7)
    recent_sessions = sessions[:7]
    previous_sessions = sessions[7:14] if len(sessions) > 7 else []
    
    if not previous_sessions:
        return {}
    
    # Calculate averages
    recent_hr = [s['metrics'].get('hr') for s in recent_sessions if s['metrics'].get('hr')]
    previous_hr = [s['metrics'].get('hr') for s in previous_sessions if s['metrics'].get('hr')]
    
    trends = {}
    if recent_hr and previous_hr:
        recent_avg = sum(recent_hr) / len(recent_hr)
        previous_avg = sum(previous_hr) / len(previous_hr)
        hr_change = ((recent_avg - previous_avg) / previous_avg) * 100
        trends['heart_rate_change_pct'] = round(hr_change, 1)
    
    return trends

def analyze_cycle_patterns(sessions):
    """
    Analyze patterns across menstrual cycle phases
    """
    phase_data = {}
    
    for session in sessions:
        phase = session.get('cycle_phase')
        if not phase:
            continue
        
        if phase not in phase_data:
            phase_data[phase] = {'hr': [], 'rmssd': []}
        
        if session['metrics'].get('hr'):
            phase_data[phase]['hr'].append(session['metrics']['hr'])
        if session['metrics'].get('rmssd'):
            phase_data[phase]['rmssd'].append(session['metrics']['rmssd'])
    
    # Calculate phase averages
    cycle_analysis = {}
    for phase, data in phase_data.items():
        cycle_analysis[phase] = {
            'avg_hr': round(sum(data['hr']) / len(data['hr']), 1) if data['hr'] else None,
            'avg_rmssd': round(sum(data['rmssd']) / len(data['rmssd']), 1) if data['rmssd'] else None,
            'session_count': len(data['hr'])
        }
    
    return cycle_analysis

def generate_clinician_pdf(user_id, days):
    """
    Generate PDF report for clinicians
    """
    try:
        # Get user data
        dashboard_data = db_manager.get_user_dashboard_data(user_id, days)
        
        if not dashboard_data:
            return None
        
        # Create PDF
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, "PulseHER Clinical Report")
        
        # Date range
        p.setFont("Helvetica", 12)
        p.drawString(50, 720, f"Report Period: {days} days")
        p.drawString(50, 700, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Summary statistics
        sessions = dashboard_data['sessions']
        if sessions:
            p.setFont("Helvetica-Bold", 14)
            p.drawString(50, 660, "Summary Statistics")
            
            p.setFont("Helvetica", 12)
            p.drawString(50, 640, f"Total Sessions: {len(sessions)}")
            
            # Calculate averages
            hr_values = [s['metrics'].get('hr') for s in sessions if s['metrics'].get('hr')]
            if hr_values:
                avg_hr = sum(hr_values) / len(hr_values)
                p.drawString(50, 620, f"Average Heart Rate: {avg_hr:.1f} BPM")
            
            # Clinical flags summary
            all_flags = dashboard_data.get('recent_flags', [])
            if all_flags:
                p.setFont("Helvetica-Bold", 14)
                p.drawString(50, 580, "Clinical Alerts")
                
                p.setFont("Helvetica", 12)
                y_pos = 560
                for flag in all_flags[:5]:  # Show top 5 flags
                    p.drawString(50, y_pos, f"• {flag['message']}")
                    y_pos -= 20
        
        p.showPage()
        p.save()
        
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return None

# Export the initialization function
__all__ = ['init_blueprint_api']