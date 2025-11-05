import React, { useState, useEffect, useRef } from 'react';
import './PPGMonitor.css';
import SaveSessionModal from './SaveSessionModal';

const PPGMonitor = ({ onMeasurementComplete, user }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [currentMetrics, setCurrentMetrics] = useState(null);
  const [sessionAnalysis, setSessionAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [ppgBuffer, setPpgBuffer] = useState([]); // for waveform time axis
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const API_BASE = 'http://localhost:5000/api';

  // Start camera and PPG session
  const startPPGSession = async () => {
    try {
      setError(null);
      
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });
      
      setCameraStream(stream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Start PPG session on backend
      const response = await fetch(`${API_BASE}/ppg/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      const data = await response.json();
      
      if (data.success) {
        setSessionId(data.session_id);
        setIsRecording(true);
        setStartTime(Date.now());
        setElapsedTime(0);
        setPpgBuffer([]);
        // Start capturing frames
        startFrameCapture();
      } else {
        throw new Error(data.error);
      }
      
    } catch (err) {
      setError(`Failed to start PPG session: ${err.message}`);
      console.error('PPG session error:', err);
    }
  };

  // Capture and send frames to backend
  const startFrameCapture = () => {
    intervalRef.current = setInterval(async () => {
      if (videoRef.current && canvasRef.current && isRecording) {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        // Set canvas size
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        // Draw current frame to canvas
        context.drawImage(videoRef.current, 0, 0);
        // Convert to base64
        const frameData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        try {
          // Send frame to backend
          const response = await fetch(`${API_BASE}/ppg/upload-frame`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              frame_data: frameData
            })
          });
          const result = await response.json();
          if (result.success) {
            setCurrentMetrics(result.data.metrics);
            setPpgBuffer(prev => [...prev, result.data.ppg_value]);
          }
        } catch (err) {
          console.error('Frame upload error:', err);
        }
        // Update elapsed time using wall clock
        if (startTime) {
          setElapsedTime(((Date.now() - startTime) / 1000).toFixed(1));
        }
      }
    }, 100); // Capture every 100ms (10 fps)
  };
  // Generate time labels for waveform using real elapsed time
  const getTimeLabels = () => {
    if (ppgBuffer.length < 2 || elapsedTime === 0) return [];
    const timePerSample = elapsedTime / ppgBuffer.length;
    return ppgBuffer.map((_, i) => (i * timePerSample).toFixed(2));
  };
      {/* Real-time Timer Display */}
      {isRecording && (
        <div className="timer-display">
          <span>Elapsed Time: {elapsedTime} s</span>
        </div>
      )}

      {/* Signal Quality Status Panel */}
      {isRecording && currentMetrics && (
        <div className="signal-quality-panel" style={{marginTop: '1em', padding: '0.75em', border: '1px solid #ccc', borderRadius: '8px', background: '#f8f8fa', maxWidth: 340}}>
          <div style={{fontWeight: 'bold', marginBottom: 4}}>Signal Quality: {(() => {
            if (currentMetrics.signal_quality === 'good' || currentMetrics.status === 'ok') return 'Good';
            if (currentMetrics.signal_quality === 'too_short' || currentMetrics.status === 'too_short') return 'No Data';
            return 'Weak';
          })()}</div>
          <div>AC/DC: {currentMetrics.acdc ? Number(currentMetrics.acdc).toFixed(3) : '--'}</div>
          <div>SNR: {currentMetrics.snr ? Number(currentMetrics.snr).toFixed(2) : '--'}</div>
          <div>BandPowerRatio(0.8–3Hz): {currentMetrics['band_power_ratio_0.8-3.0Hz'] !== undefined ? Number(currentMetrics['band_power_ratio_0.8-3.0Hz']).toFixed(2) : '--'}</div>
          <div style={{marginTop: 6}}>
            Decision: <span style={{color:
              currentMetrics.decision === 'likely_pulse' ? '#0a0' :
              currentMetrics.decision === 'likely_noise' ? '#b00' : '#888', fontWeight: 'bold'}}>
              {currentMetrics.decision ? currentMetrics.decision.replace('_', ' ') : '--'}
            </span>
          </div>
          {currentMetrics.status_advice && (
            <div style={{color: currentMetrics.decision === 'likely_noise' ? '#b00' : '#333', marginTop: 6, fontSize: '0.97em'}}>
              Advice: {currentMetrics.status_advice}
            </div>
          )}
        </div>
      )}
      {/* Only run advanced metrics if likely_pulse, else prompt user to re-record */}
      {isRecording && currentMetrics && currentMetrics.decision === 'likely_noise' && (
        <div style={{marginTop: '1em', color: '#b00', fontWeight: 500}}>
          Signal too noisy for analysis. Please re-record with better lighting and a steady finger.
        </div>
      )}

  // Stop PPG session
  const stopPPGSession = async () => {
    try {
      setIsRecording(false);
      setElapsedTime(0);
      setPpgBuffer([]); // Clear waveform buffer so it doesn't keep updating
      // Stop frame capture
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      // Stop camera stream
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        setCameraStream(null);
      }
      // Stop session on backend
      const response = await fetch(`${API_BASE}/ppg/stop-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      const data = await response.json();
      if (data.success) {
        // Get session analysis
        await getSessionAnalysis();
      }
    } catch (err) {
      setError(`Failed to stop PPG session: ${err.message}`);
      console.error('Stop session error:', err);
    }
  };

  // Get session analysis
  const getSessionAnalysis = async () => {
    try {
      const response = await fetch(`${API_BASE}/ppg/analyze-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      const data = await response.json();
      
      if (data.success) {
        setSessionAnalysis(data.analysis);
        setShowSaveModal(true);
        // Call parent component's callback to save to Firebase
        if (onMeasurementComplete) {
          onMeasurementComplete(data.analysis);
        }
      }
  // Save session to longitudinal tracking
  const handleSaveSession = async () => {
    try {
      // Send sessionAnalysis to backend for longitudinal tracking
      await fetch(`${API_BASE}/ppg/save-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis: sessionAnalysis, user })
      });
      setShowSaveModal(false);
      // Optionally show a confirmation or reset UI
    } catch (err) {
      setError('Failed to save session.');
    }
  };

  // Delete session (just clear analysis)
  const handleDeleteSession = () => {
    setSessionAnalysis(null);
    setShowSaveModal(false);
  };
      
    } catch (err) {
      console.error('Analysis error:', err);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [cameraStream]);

  const renderMetrics = () => {
    if (!currentMetrics) return null;

    const { heart_rate, signal_quality, status, physiological_metrics } = currentMetrics;

    return (
      <div className="ppg-metrics">
        <div className="metric-card">
          <h3>Status</h3>
          <p className={`status ${status}`}>{status}</p>
        </div>

        {heart_rate && (
          <div className="metric-card">
            <h3>Heart Rate</h3>
            <p className="heart-rate">{Math.round(heart_rate)} BPM</p>
          </div>
        )}

        <div className="metric-card">
          <h3>Signal Quality</h3>
          <p className={`quality ${signal_quality}`}>{signal_quality}</p>
        </div>

        {physiological_metrics && (
          <div className="metric-card advanced-metrics">
            <h3>Advanced Metrics</h3>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li><b>SDNN:</b> {physiological_metrics.hrv_sdnn ? physiological_metrics.hrv_sdnn.toFixed(1) : 0} ms</li>
              <li><b>RMSSD:</b> {physiological_metrics.hrv_rmssd ? physiological_metrics.hrv_rmssd.toFixed(1) : 0} ms</li>
              <li><b>LF/HF Ratio:</b> {physiological_metrics.lf_hf_ratio ? physiological_metrics.lf_hf_ratio.toFixed(2) : 0}</li>
              <li><b>pNN50:</b> {physiological_metrics.pnn50 ? physiological_metrics.pnn50.toFixed(1) : 0} %</li>
              <li><b>Valid Beats:</b> {physiological_metrics.valid_intervals || 0}</li>
            </ul>
          </div>
        )}
      </div>
    );
  };

  const renderAnalysis = () => {
    if (!sessionAnalysis) return null;

    const { heart_rate_metrics, risk_assessment, recommendations } = sessionAnalysis;

    return (
      <div className="session-analysis">
        <h2>Session Analysis</h2>
        
        <div className="analysis-section">
          <h3>Heart Rate Metrics</h3>
          <div className="metrics-grid">
            <div>Mean: {heart_rate_metrics.mean} BPM</div>
            <div>Range: {heart_rate_metrics.range} BPM</div>
            <div>Variability: {heart_rate_metrics.std_deviation}</div>
          </div>
        </div>

        <div className="analysis-section">
          <h3>Risk Assessment</h3>
          <div className={`risk-level ${risk_assessment.risk_level.toLowerCase().replace(' ', '-')}`}>
            {risk_assessment.risk_level} ({risk_assessment.risk_percentage}%)
          </div>
        </div>

        <div className="analysis-section">
          <h3>Recommendations</h3>
          <ul className="recommendations">
            {recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  return (
    <div className="ppg-monitor">
      <div className="ppg-header">
        <h1>🩸 PulseHER Heart Rate Monitor</h1>
        <p>Place your finger over the camera lens with flash enabled</p>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      <div className="ppg-content">
        {/* Save/Delete Modal */}
        {showSaveModal && (
          <SaveSessionModal
            onSave={handleSaveSession}
            onDelete={handleDeleteSession}
            onClose={() => setShowSaveModal(false)}
          />
        )}
        {/* Camera Section */}
        <div className="camera-section">
          <video
            ref={videoRef}
            autoPlay
            muted
            className="camera-feed"
            style={{ display: isRecording ? 'block' : 'none' }}
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          
          {!isRecording && !sessionAnalysis && (
            <div className="start-prompt">
              <p>📱 Ready to start PPG monitoring</p>
              <button onClick={startPPGSession} className="start-button">
                Start Monitoring
              </button>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="ppg-controls">
          {isRecording ? (
            <button onClick={stopPPGSession} className="stop-button">
              Stop Recording
            </button>
          ) : (
            sessionAnalysis && (
              <button onClick={() => setSessionAnalysis(null)} className="reset-button">
                New Session
              </button>
            )
          )}
        </div>

        {/* Real-time Metrics */}
        {isRecording && renderMetrics()}

        {/* Session Analysis */}
        {sessionAnalysis && renderAnalysis()}
      </div>

      {/* Instructions */}
      <div className="ppg-instructions">
        <h3>📋 Instructions:</h3>
        <ol>
          <li>Place your fingertip gently over the rear camera lens</li>
          <li>Ensure the camera flash is on for better signal</li>
          <li>Keep your finger steady for 30-60 seconds</li>
          <li>Avoid pressing too hard to maintain blood flow</li>
          <li>Stay still and breathe normally during recording</li>
        </ol>
      </div>
    </div>
  );
};

export default PPGMonitor;