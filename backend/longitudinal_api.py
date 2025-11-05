"""
Longitudinal Tracking API Endpoints
=================================

API integration for longitudinal data tracking system.
Seamlessly integrates with existing PulseHER app flow without changing UI.

Endpoints:
- POST /api/longitudinal/profile - Update user profile
- POST /api/longitudinal/metrics - Save metrics batch
- POST /api/longitudinal/session/start - Start PPG session
- POST /api/longitudinal/session/complete - Complete PPG session
- GET  /api/longitudinal/dashboard/<user_id> - Get dashboard data
- GET  /api/longitudinal/trends/<user_id> - Get trend analysis
- GET  /api/longitudinal/export/<user_id> - Export user data

Author: PulseHER Team
Version: 1.0
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import traceback

# Import the longitudinal tracking system
try:
    from longitudinal_tracking import (
        get_longitudinal_manager, 
        initialize_longitudinal_tracking,
        save_ppg_session_data,
        save_user_metrics,
        update_user_profile,
        get_user_dashboard_data
    )
    LONGITUDINAL_AVAILABLE = True
    print("[OK] Longitudinal tracking imported successfully")
except ImportError as e:
    print(f"[ERROR] Longitudinal tracking not available: {e}")
    LONGITUDINAL_AVAILABLE = False

def init_longitudinal_api(app):
    """Initialize longitudinal tracking API routes"""
    
    if not LONGITUDINAL_AVAILABLE:
        print("[WARN] Longitudinal API disabled - tracking module not available")
        return
    
    # Initialize the tracking system
    manager = initialize_longitudinal_tracking()
    if not manager:
        print("[WARN] Longitudinal API disabled - initialization failed")
        return
    
    @app.route('/api/longitudinal/profile', methods=['POST'])
    def update_profile():
        """Update or create user profile with cycle and health information"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            
            if not user_id:
                return jsonify({
                    'success': False,
                    'error': 'user_id is required'
                }), 400
            
            # Extract profile data
            profile_data = {k: v for k, v in data.items() if k != 'user_id'}
            
            # Update profile
            profile = update_user_profile(user_id, profile_data)
            
            if profile:
                return jsonify({
                    'success': True,
                    'message': 'Profile updated successfully',
                    'profile': {
                        'user_id': profile.user_id,
                        'cycle_length_days': profile.cycle_length_days,
                        'last_period_start': profile.last_period_start,
                        'created_at': profile.created_at
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to update profile'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/metrics', methods=['POST'])
    def save_metrics():
        """Save batch of user metrics to longitudinal tracking"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            metrics = data.get('metrics', {})
            source = data.get('source', 'manual')
            
            if not user_id:
                return jsonify({
                    'success': False,
                    'error': 'user_id is required'
                }), 400
            
            if not metrics:
                return jsonify({
                    'success': False,
                    'error': 'metrics data is required'
                }), 400
            
            # Save metrics
            saved_metrics = save_user_metrics(user_id, metrics, source)
            
            if saved_metrics:
                return jsonify({
                    'success': True,
                    'message': f'Saved {len(saved_metrics)} metrics to longitudinal tracking',
                    'metrics_saved': [m.metric_name for m in saved_metrics],
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save metrics'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/session/start', methods=['POST'])
    def start_session():
        """Start a new PPG measurement session in longitudinal tracking"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            session_id = data.get('session_id')
            
            if not user_id or not session_id:
                return jsonify({
                    'success': False,
                    'error': 'user_id and session_id are required'
                }), 400
            
            manager = get_longitudinal_manager()
            session = manager.start_ppg_session(user_id, session_id)
            
            return jsonify({
                'success': True,
                'message': 'PPG session started in longitudinal tracking',
                'session': {
                    'session_id': session.session_id,
                    'user_id': session.user_id,
                    'start_timestamp': session.start_timestamp,
                    'cycle_info': session.cycle_info
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/session/complete', methods=['POST'])
    def complete_session():
        """Complete a PPG session with final results"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            session_id = data.get('session_id')
            session_results = data.get('session_results', {})
            
            if not user_id or not session_id:
                return jsonify({
                    'success': False,
                    'error': 'user_id and session_id are required'
                }), 400
            
            # Save session data
            session = save_ppg_session_data(user_id, session_id, session_results)
            
            if session:
                return jsonify({
                    'success': True,
                    'message': 'PPG session completed and saved to longitudinal tracking',
                    'session': {
                        'session_id': session.session_id,
                        'start_timestamp': session.start_timestamp,
                        'end_timestamp': session.end_timestamp,
                        'duration_seconds': session.duration_seconds,
                        'signal_quality': session.signal_quality
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to complete session'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/dashboard/<user_id>', methods=['GET'])
    def get_dashboard(user_id):
        """Get comprehensive dashboard data for user"""
        try:
            days_back = request.args.get('days', 30, type=int)
            
            dashboard_data = get_user_dashboard_data(user_id, days_back)
            
            return jsonify({
                'success': True,
                'data': dashboard_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/trends/<user_id>', methods=['GET'])
    def get_trends(user_id):
        """Get trend analysis for specific metrics"""
        try:
            metric_names = request.args.getlist('metrics')
            days_back = request.args.get('days', 90, type=int)
            
            if not metric_names:
                metric_names = ['heart_rate', 'rmssd', 'sdnn', 'stress_score']
            
            manager = get_longitudinal_manager()
            
            trends = {}
            for metric_name in metric_names:
                # Overall trend
                overall_baseline = manager.calculate_user_baselines(user_id, metric_name)
                
                # Phase-specific trends
                phase_baselines = {}
                for phase in ['menstrual', 'follicular', 'ovulation', 'luteal']:
                    phase_baseline = manager.calculate_user_baselines(user_id, metric_name, phase)
                    if phase_baseline:
                        phase_baselines[phase] = phase_baseline
                
                if overall_baseline:
                    trends[metric_name] = {
                        'overall': overall_baseline,
                        'by_cycle_phase': phase_baselines
                    }
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'date_range': {
                    'start': (datetime.now() - timedelta(days=days_back)).isoformat(),
                    'end': datetime.now().isoformat()
                },
                'trends': trends
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/metrics/<user_id>', methods=['GET'])
    def get_user_metrics(user_id):
        """Get user's historical metrics"""
        try:
            days_back = request.args.get('days', 30, type=int)
            metric_names = request.args.getlist('metrics')  # Optional filter
            
            manager = get_longitudinal_manager()
            
            # Get metrics
            if metric_names:
                metrics = manager.get_user_metrics(user_id, metric_names, days_back)
            else:
                metrics = manager.get_user_metrics(user_id, None, days_back)
            
            # Format for JSON response
            metrics_data = []
            for metric in metrics:
                metrics_data.append({
                    'timestamp': metric.timestamp,
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'source': metric.source,
                    'cycle_phase': metric.cycle_phase,
                    'cycle_day': metric.cycle_day,
                    'session_id': metric.session_id,
                    'quality_score': metric.quality_score
                })
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'total_metrics': len(metrics_data),
                'date_range': {
                    'start': (datetime.now() - timedelta(days=days_back)).isoformat(),
                    'end': datetime.now().isoformat()
                },
                'metrics': metrics_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/export/<user_id>', methods=['GET'])
    def export_data(user_id):
        """Export all user data for research or backup"""
        try:
            anonymize = request.args.get('anonymize', 'false').lower() == 'true'
            
            manager = get_longitudinal_manager()
            export_data = manager.export_user_data(user_id, anonymize)
            
            return jsonify({
                'success': True,
                'export_data': export_data,
                'anonymized': anonymize,
                'export_timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/cleanup/<user_id>', methods=['POST'])
    def cleanup_old_data(user_id):
        """Clean up old data beyond retention period"""
        try:
            days_to_keep = request.args.get('days_to_keep', 365, type=int)
            
            manager = get_longitudinal_manager()
            manager.cleanup_old_data(user_id, days_to_keep)
            
            return jsonify({
                'success': True,
                'message': f'Cleaned up data older than {days_to_keep} days for user {user_id}'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    @app.route('/api/longitudinal/status', methods=['GET'])
    def get_status():
        """Get longitudinal tracking system status"""
        try:
            manager = get_longitudinal_manager()
            
            total_users = len(manager.user_profiles)
            total_metrics = sum(len(metrics) for metrics in manager.user_metrics.values())
            total_sessions = sum(len(sessions) for sessions in manager.user_sessions.values())
            
            return jsonify({
                'success': True,
                'status': 'active',
                'statistics': {
                    'total_users': total_users,
                    'total_metrics_tracked': total_metrics,
                    'total_ppg_sessions': total_sessions,
                    'data_directory': manager.data_directory
                },
                'system_info': {
                    'version': '1.0',
                    'initialized': True,
                    'data_persistence': 'json_files'
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }), 500
    
    print("[OK] Longitudinal tracking API endpoints initialized successfully")
    print("[INFO] Available endpoints:")
    print("  - POST /api/longitudinal/profile")
    print("  - POST /api/longitudinal/metrics") 
    print("  - POST /api/longitudinal/session/start")
    print("  - POST /api/longitudinal/session/complete")
    print("  - GET  /api/longitudinal/dashboard/<user_id>")
    print("  - GET  /api/longitudinal/trends/<user_id>")
    print("  - GET  /api/longitudinal/metrics/<user_id>")
    print("  - GET  /api/longitudinal/export/<user_id>")
    print("  - POST /api/longitudinal/cleanup/<user_id>")
    print("  - GET  /api/longitudinal/status")
