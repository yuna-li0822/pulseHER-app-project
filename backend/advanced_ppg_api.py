"""
Enhanced PPG API for PulseHER v4.0
Advanced HRV Analytics & Cycle-Aware Processing
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import json
import base64
import cv2
from datetime import datetime, timedelta
import io
from advanced_ppg_processor import AdvancedPPGProcessor
import logging

# Initialize advanced PPG processor
advanced_processor = AdvancedPPGProcessor()
ppg_sessions = {}  # Store active sessions
user_baselines = {}  # Store user baseline data

def init_advanced_ppg_routes(app):
    """Initialize advanced PPG routes for the Flask app"""
    
    @app.route('/api/ppg/start-advanced-session', methods=['POST'])
    def start_advanced_ppg_session():
        """Start advanced PPG monitoring session with cycle info"""
        try:
            data = request.get_json()
            
            # Extract user info
            user_id = data.get('user_id', 'anonymous')
            last_period_start = data.get('last_period_start')  # YYYY-MM-DD
            cycle_length = data.get('cycle_length', 28)
            duration_seconds = data.get('duration_seconds', 60)
            
            # Create session
            session_id = f"ppg_{int(datetime.now().timestamp())}"
            
            ppg_sessions[session_id] = {
                'session_id': session_id,
                'user_id': user_id,
                'start_time': datetime.now().isoformat(),
                'last_period_start': last_period_start,
                'cycle_length': cycle_length,
                'duration_seconds': duration_seconds,
                'rr_intervals': [],
                'frames_processed': 0,
                'status': 'active'
            }
            
            # Get cycle info
            cycle_info = {}
            if last_period_start:
                cycle_info = advanced_processor.estimate_cycle_phase(last_period_start, cycle_length)
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "cycle_info": cycle_info,
                "message": "Advanced PPG session started successfully"
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to start session: {str(e)}"
            }), 500
    
    @app.route('/api/ppg/upload-advanced-frame', methods=['POST'])
    def upload_advanced_frame():
        """Process PPG frame with advanced analysis"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            frame_data = data.get('frame_data')  # Base64 encoded image
            timestamp = data.get('timestamp', datetime.now().timestamp())
            
            if session_id not in ppg_sessions:
                return jsonify({"success": False, "error": "Invalid session"}), 400
            
            session = ppg_sessions[session_id]
            
            # Decode frame
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            image_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"success": False, "error": "Invalid frame data"}), 400
            
            # Extract PPG signal from frame
            ppg_value, quality_metrics = extract_ppg_from_frame(frame)
            
            # Simple beat detection (in practice, you'd use more sophisticated algorithms)
            if session['frames_processed'] > 10:  # Need some frames for comparison
                rr_interval = detect_rr_interval(ppg_value, session, timestamp)
                if rr_interval:
                    session['rr_intervals'].append(rr_interval)
            
            session['frames_processed'] += 1
            
            # Real-time heart rate estimation
            current_hr = 0
            if len(session['rr_intervals']) >= 3:
                recent_rr = session['rr_intervals'][-3:]
                avg_rr = np.mean(recent_rr)
                current_hr = 60000 / avg_rr if avg_rr > 0 else 0
            
            return jsonify({
                "success": True,
                "heart_rate": current_hr,
                "signal_quality": quality_metrics.get('quality', 'processing'),
                "rr_count": len(session['rr_intervals']),
                "frames_processed": session['frames_processed']
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Frame processing error: {str(e)}"
            }), 500
    
    @app.route('/api/ppg/analyze-advanced-session/<session_id>', methods=['POST'])
    def analyze_advanced_session(session_id):
        """Complete advanced analysis of PPG session"""
        try:
            if session_id not in ppg_sessions:
                return jsonify({"success": False, "error": "Session not found"}), 404
            
            session = ppg_sessions[session_id]
            rr_intervals = session.get('rr_intervals', [])
            
            if len(rr_intervals) < 5:
                return jsonify({
                    "success": False,
                    "error": "Insufficient data for analysis"
                }), 400
            
            # Get user baselines if available
            user_id = session.get('user_id')
            baselines = user_baselines.get(user_id, {})
            
            # Perform complete analysis
            analysis_results = advanced_processor.process_ppg_session(
                rr_intervals=rr_intervals,
                last_period_start=session.get('last_period_start'),
                cycle_length=session.get('cycle_length', 28),
                user_baselines=baselines
            )
            
            # Update user baselines (simple moving average)
            update_user_baselines(user_id, analysis_results)
            
            # Mark session as complete
            session['status'] = 'completed'
            session['analysis_results'] = analysis_results
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "analysis": analysis_results
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }), 500
    
    @app.route('/api/ppg/user-profile', methods=['POST'])
    def update_user_profile():
        """Update user profile with cycle and health info"""
        try:
            data = request.get_json()
            user_id = data.get('user_id', 'anonymous')
            
            profile = {
                'age': data.get('age'),
                'weight_kg': data.get('weight_kg'),
                'height_cm': data.get('height_cm'),
                'cycle_length_days': data.get('cycle_length_days', 28),
                'last_period_start': data.get('last_period_start'),
                'contraception_type': data.get('contraception_type'),
                'medical_conditions': data.get('medical_conditions', []),
                'medications': data.get('medications', []),
                'lifestyle_factors': {
                    'sleep_hours': data.get('sleep_hours', 7),
                    'caffeine_cups': data.get('caffeine_cups', 1),
                    'exercise_minutes': data.get('exercise_minutes', 30),
                    'stress_level': data.get('stress_level', 3)  # 1-5 scale
                },
                'updated_at': datetime.now().isoformat()
            }
            
            # Store profile (in production, use proper database)
            if 'user_profiles' not in globals():
                global user_profiles
                user_profiles = {}
            
            user_profiles[user_id] = profile
            
            return jsonify({
                "success": True,
                "message": "Profile updated successfully"
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Profile update failed: {str(e)}"
            }), 500
    
    @app.route('/api/ppg/dashboard/<user_id>', methods=['GET'])
    def get_advanced_dashboard(user_id):
        """Get advanced dashboard data with cycle-aware analytics"""
        try:
            # Get query parameters
            days = int(request.args.get('days', 7))
            
            # Get user's recent sessions (mock data for now)
            recent_sessions = get_user_sessions(user_id, days)
            
            # Calculate trends and insights
            dashboard_data = calculate_dashboard_metrics(recent_sessions, user_id)
            
            return jsonify({
                "success": True,
                "dashboard": dashboard_data
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Dashboard error: {str(e)}"
            }), 500
    
    @app.route('/api/ppg/export-clinician/<user_id>', methods=['GET'])
    def export_advanced_clinician_report(user_id):
        """Generate clinician PDF report"""
        try:
            # Generate PDF report (simplified version)
            pdf_buffer = generate_clinician_pdf(user_id)
            
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'pulseher_report_{user_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
            )
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Report generation failed: {str(e)}"
            }), 500

def extract_ppg_from_frame(frame):
    """
    Extract PPG signal from camera frame
    Simplified version - in practice would use more sophisticated algorithms
    """
    try:
        # Convert to different color spaces for PPG extraction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Focus on central region where finger should be
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        roi_size = min(width, height) // 4
        
        roi = rgb_frame[
            center_y - roi_size:center_y + roi_size,
            center_x - roi_size:center_x + roi_size
        ]
        
        if roi.size == 0:
            return 0, {'quality': 'poor'}
        
        # Extract green channel (best for PPG)
        green_channel = roi[:, :, 1]
        
        # Calculate mean intensity
        mean_intensity = np.mean(green_channel)
        
        # Calculate quality metrics
        std_intensity = np.std(green_channel)
        brightness = mean_intensity / 255.0
        
        # Quality assessment
        if brightness > 0.8:
            quality = 'saturated'
        elif brightness < 0.1:
            quality = 'too_dark'
        elif std_intensity < 10:
            quality = 'good'
        else:
            quality = 'fair'
        
        return mean_intensity, {
            'quality': quality,
            'brightness': brightness,
            'std_intensity': std_intensity
        }
        
    except Exception as e:
        logging.error(f"PPG extraction error: {e}")
        return 0, {'quality': 'error'}

def detect_rr_interval(current_ppg, session, timestamp):
    """
    Simple RR interval detection
    In practice, would use more sophisticated peak detection
    """
    try:
        # Store PPG values in session
        if 'ppg_values' not in session:
            session['ppg_values'] = []
            session['ppg_timestamps'] = []
        
        session['ppg_values'].append(current_ppg)
        session['ppg_timestamps'].append(timestamp)
        
        # Keep only recent values (last 10 seconds)
        cutoff_time = timestamp - 10000  # 10 seconds in ms
        valid_indices = [i for i, t in enumerate(session['ppg_timestamps']) if t >= cutoff_time]
        
        if len(valid_indices) < len(session['ppg_values']):
            session['ppg_values'] = [session['ppg_values'][i] for i in valid_indices]
            session['ppg_timestamps'] = [session['ppg_timestamps'][i] for i in valid_indices]
        
        # Simple peak detection (needs improvement)
        if len(session['ppg_values']) >= 10:
            values = np.array(session['ppg_values'][-10:])
            timestamps = np.array(session['ppg_timestamps'][-10:])
            
            # Find peaks
            mean_val = np.mean(values)
            peaks = []
            for i in range(1, len(values) - 1):
                if values[i] > values[i-1] and values[i] > values[i+1] and values[i] > mean_val:
                    peaks.append(i)
            
            # Calculate RR interval if we have recent peaks
            if len(peaks) >= 2:
                last_peak_time = timestamps[peaks[-1]]
                prev_peak_time = timestamps[peaks[-2]]
                rr_interval = last_peak_time - prev_peak_time
                
                # Sanity check (250ms to 2000ms)
                if 250 <= rr_interval <= 2000:
                    return rr_interval
        
        return None
        
    except Exception as e:
        logging.error(f"RR detection error: {e}")
        return None

def update_user_baselines(user_id, analysis_results):
    """Update user's baseline metrics with exponential moving average"""
    if user_id not in user_baselines:
        user_baselines[user_id] = {}
    
    # Get current baselines
    baselines = user_baselines[user_id]
    alpha = 0.1  # Smoothing factor
    
    # Update key metrics
    metrics_to_track = ['mean_hr_bpm', 'rmssd_ms', 'sdnn_ms', 'lf_hf_ratio']
    
    for metric in metrics_to_track:
        if metric in analysis_results:
            current_value = analysis_results[metric]
            if metric in baselines:
                # Exponential moving average
                baselines[metric] = alpha * current_value + (1 - alpha) * baselines[metric]
            else:
                # First measurement
                baselines[metric] = current_value

def get_user_sessions(user_id, days):
    """Get user's recent sessions (mock implementation)"""
    # In production, query database
    return []

def calculate_dashboard_metrics(sessions, user_id):
    """Calculate dashboard metrics and trends"""
    return {
        'summary': {
            'avg_hr': 72,
            'avg_rmssd': 35,
            'sessions_count': len(sessions),
            'trend_hr': 'stable',
            'trend_hrv': 'improving'
        },
        'cycle_awareness': {
            'current_phase': 'follicular',
            'expected_hr_range': [65, 75],
            'expected_hrv_range': [30, 45]
        },
        'alerts': []
    }

def generate_clinician_pdf(user_id):
    """Generate PDF report for clinicians (simplified)"""
    # In production, use ReportLab or similar
    from io import BytesIO
    
    # Mock PDF content
    pdf_content = f"PulseHER Clinical Report for User {user_id}\nGenerated: {datetime.now()}"
    
    buffer = BytesIO()
    buffer.write(pdf_content.encode('utf-8'))
    buffer.seek(0)
    
    return buffer

# Initialize logging
logging.basicConfig(level=logging.INFO)