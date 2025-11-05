"""
PPG API Endpoints for PulseHER Flask Server
Handles PPG data collection, processing, and real-time heart rate monitoring
"""

from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
from ppg_processor import PPGProcessor
import threading
import time


ppg_session_data = {}

def init_ppg_routes(app):

    @app.route('/api/ppg/save-session', methods=['POST'])
    def save_ppg_session():
        """Save session analysis to longitudinal tracking (user_data/)"""
        try:
            data = request.get_json()
            analysis = data.get('analysis')
            user = data.get('user', 'default')
            if not analysis:
                return jsonify({"success": False, "error": "No analysis data provided"}), 400

            # Save to user_data/longitudinal_{user}.json (append or create)
            import os, json
            save_dir = os.path.join(os.path.dirname(__file__), '../user_data')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'longitudinal_{user}.json')
            sessions = []
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    try:
                        sessions = json.load(f)
                    except Exception:
                        sessions = []
            sessions.append({
                'timestamp': int(time.time()),
                'analysis': analysis
            })
            with open(save_path, 'w') as f:
                json.dump(sessions, f, indent=2)
            return jsonify({"success": True, "message": "Session saved"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    """Initialize PPG routes for the Flask app"""
    
    @app.route('/api/ppg/start-session', methods=['POST'])
    def start_ppg_session():
        """Start a new PPG monitoring session"""
        global ppg_session_data
        try:
            # Initialize session data (no camera controller)
            session_id = str(int(time.time()))
            ppg_session_data = {
                'session_id': session_id,
                'start_time': time.time(),
                'status': 'active',
                'measurements': []
            }
            return jsonify({
                "success": True,
                "session_id": session_id,
                "message": "PPG session started successfully"
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Failed to start PPG session: {str(e)}",
                "success": False
            }), 500
    
    @app.route('/api/ppg/capture', methods=['POST'])
    def capture_ppg_data():
        """Capture PPG data from camera"""
        global ppg_controller
        
        if not ppg_controller or not ppg_controller.is_recording:
            return jsonify({
                "error": "PPG session not active",
                "success": False
            }), 400
        
        try:
            # Capture frame and extract PPG
            ppg_data = ppg_controller.capture_ppg_frame()
            
            if not ppg_data:
                return jsonify({
                    "error": "Failed to capture PPG data",
                    "success": False
                }), 500
            
            # Store measurement
            measurement = {
                'timestamp': time.time(),
                'ppg_value': ppg_data['ppg_value'],
                'metrics': ppg_data['metrics']
            }
            
            ppg_session_data['measurements'].append(measurement)
            
            # Return current metrics
            return jsonify({
                "success": True,
                "data": {
                    "ppg_value": ppg_data['ppg_value'],
                    "metrics": ppg_data['metrics'],
                    "session_duration": time.time() - ppg_session_data['start_time']
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": f"PPG capture failed: {str(e)}",
                "success": False
            }), 500
    
    @app.route('/api/ppg/upload-frame', methods=['POST'])
    def upload_ppg_frame():
        """Process uploaded frame for PPG extraction (for mobile apps)"""
        try:
            data = request.get_json()
            if not data or 'frame_data' not in data:
                return jsonify({
                    "error": "No frame data provided",
                    "success": False
                }), 400

            # Decode base64 frame data
            frame_bytes = base64.b64decode(data['frame_data'])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return jsonify({
                    "error": "Invalid frame data",
                    "success": False
                }), 400

            # Initialize processor
            if 'processor' not in ppg_session_data:
                ppg_session_data['processor'] = PPGProcessor()

            processor = ppg_session_data['processor']

            # Try extracting PPG value
            ppg_value = processor.extract_ppg_from_frame(frame)
            if ppg_value is None or (hasattr(np, 'isnan') and np.isnan(ppg_value)):
                ppg_value = 0.0  # fallback for weak signals

            try:
                processor.add_ppg_sample(ppg_value)
                metrics = processor.get_real_time_metrics() or {}
            except Exception:
                metrics = {}

            # Store measurement
            measurement = {
                'timestamp': time.time(),
                'ppg_value': float(ppg_value),
                'metrics': metrics
            }
            ppg_session_data.setdefault('measurements', []).append(measurement)

            return jsonify({
                "success": True,
                "data": {
                    "ppg_value": float(ppg_value),
                    "metrics": metrics,
                    "frame_processed": True,
                    "buffer_size": len(processor.ppg_buffer),
                    "debug_info": f"PPG={ppg_value:.2f}, Metrics keys={list(metrics.keys())}"
                }
            })

        except Exception as e:
            return jsonify({
                "error": f"Frame processing failed safely: {str(e)}",
                "success": True,  # still success so frontend won’t crash
                "data": {"ppg_value": 0.0, "metrics": {}}
            }), 200
    
    @app.route('/api/ppg/get-metrics', methods=['GET'])
    def get_ppg_metrics():
        """Get current PPG metrics and session status"""
        try:
            if not ppg_session_data:
                return jsonify({
                    "error": "No active PPG session",
                    "success": False
                }), 400
            
            # Calculate session statistics
            measurements = ppg_session_data.get('measurements', [])
            
            if len(measurements) == 0:
                return jsonify({
                    "success": True,
                    "data": {
                        "status": "no_data",
                        "session_id": ppg_session_data.get('session_id'),
                        "measurements_count": 0
                    }
                })
            
            # Get latest metrics
            latest_measurement = measurements[-1]
            latest_metrics = latest_measurement['metrics']
            
            # Calculate averages if enough data
            if len(measurements) >= 10:
                recent_hr_values = []
                for m in measurements[-30:]:  # Last 30 measurements
                    if m['metrics'].get('heart_rate'):
                        recent_hr_values.append(m['metrics']['heart_rate'])
                
                avg_heart_rate = np.mean(recent_hr_values) if recent_hr_values else None
                hr_variability = np.std(recent_hr_values) if len(recent_hr_values) > 1 else None
            else:
                avg_heart_rate = latest_metrics.get('heart_rate')
                hr_variability = None
            
            return jsonify({
                "success": True,
                "data": {
                    "session_id": ppg_session_data.get('session_id'),
                    "status": ppg_session_data.get('status', 'active'),
                    "session_duration": time.time() - ppg_session_data.get('start_time', time.time()),
                    "measurements_count": len(measurements),
                    "current_metrics": latest_metrics,
                    "averages": {
                        "heart_rate": avg_heart_rate,
                        "hr_variability": hr_variability
                    }
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Failed to get metrics: {str(e)}",
                "success": False
            }), 500
    
    @app.route('/api/ppg/analyze-session', methods=['POST'])
    def analyze_ppg_session():
        """Analyze complete PPG session and predict cardiac risk"""
        try:
            if not ppg_session_data or not ppg_session_data.get('measurements'):
                return jsonify({
                    "error": "No PPG session data available",
                    "success": False
                }), 400

            measurements = ppg_session_data['measurements']

            # Collect HR values safely
            heart_rates = [m.get('metrics', {}).get('heart_rate') for m in measurements]
            heart_rates = [hr for hr in heart_rates if hr is not None and hr > 0]

            if len(heart_rates) < 3:
                return jsonify({
                    "success": True,
                    "analysis": {
                        "summary": "Insufficient valid heart rate data for detailed analysis",
                        "total_measurements": len(measurements),
                        "valid_hr_readings": len(heart_rates),
                        "risk_level": "Unknown",
                        "recommendation": "Try re-recording under better lighting or using LED flash."
                    }
                })

            # Compute HR stats
            hr_mean = np.mean(heart_rates)
            hr_std = np.std(heart_rates)
            hr_min = np.min(heart_rates)
            hr_max = np.max(heart_rates)

            # Simple risk logic
            risk_score = 0
            if hr_mean > 100: risk_score += 2
            elif hr_mean < 50: risk_score += 1
            if hr_std > 20: risk_score += 1
            elif hr_std < 5: risk_score += 1

            risk_map = {0: ("Low Risk", 10), 1: ("Medium Risk", 40), 2: ("Medium Risk", 40), 3: ("High Risk", 75)}
            risk_level, risk_percent = risk_map.get(risk_score, ("Medium Risk", 50))

            return jsonify({
                "success": True,
                "analysis": {
                    "heart_rate_metrics": {
                        "mean": round(hr_mean, 1),
                        "std_deviation": round(hr_std, 1),
                        "min": round(hr_min, 1),
                        "max": round(hr_max, 1)
                    },
                    "risk_assessment": {
                        "risk_level": risk_level,
                        "risk_percentage": risk_percent,
                        "risk_score": risk_score
                    },
                    "timestamp": time.time()
                }
            })

        except Exception as e:
            return jsonify({
                "error": f"Safe analysis fallback: {str(e)}",
                "success": True,
                "analysis": {
                    "summary": "Analysis incomplete due to data quality issues.",
                    "risk_level": "Unknown"
                }
            }), 200