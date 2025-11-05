// =============================================
// PulseHer App.js - Web React Version with PPG + Firebase Integration
// =============================================
import React, { useState, useEffect } from 'react';
import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc, getDocs, query, orderBy, limit } from 'firebase/firestore';
import PPGMonitor from './components/PPGMonitor';
import HeartStatsCard from './components/HeartStatsCard';
import AIAssistant from './components/AIAssistant';
import './App.css';

// Firebase Configuration - Replace with your actual config
const firebaseConfig = {
  apiKey: "your-api-key",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "your-app-id"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// Main PulseHer App Component
function App() {
  const [currentView, setCurrentView] = useState('home');
  const [recentMeasurements, setRecentMeasurements] = useState([]);
  const [loading, setLoading] = useState(false);
  const [user] = useState({ id: 'demo-user-123', name: 'Demo User' });

  // Load recent measurements from Firebase
  useEffect(() => {
    loadRecentMeasurements();
  }, []);

  const loadRecentMeasurements = async () => {
    try {
      setLoading(true);
      const q = query(
        collection(db, 'measurements'), 
        orderBy('timestamp', 'desc'), 
        limit(10)
      );
      const querySnapshot = await getDocs(q);
      const measurements = [];
      querySnapshot.forEach((doc) => {
        measurements.push({ id: doc.id, ...doc.data() });
      });
      setRecentMeasurements(measurements);
    } catch (error) {
      console.error('Error loading measurements:', error);
      // For demo purposes, use mock data if Firebase fails
      setRecentMeasurements([
        {
          id: 'demo1',
          bpm: 72,
          rhythm: 'Normal',
          riskLevel: 'Low Risk',
          timestamp: { seconds: Date.now() / 1000 },
          dataSource: 'Demo'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle PPG measurement completion and save to Firebase
  const handlePPGMeasurementComplete = async (ppgAnalysis) => {
    try {
      setLoading(true);
      
      // Extract data from PPG analysis
      const heartRateMetrics = ppgAnalysis.heart_rate_metrics;
      const riskAssessment = ppgAnalysis.risk_assessment;
      
      // Save PPG measurement to Firebase
      await addDoc(collection(db, 'measurements'), {
        bpm: heartRateMetrics.mean,
        rhythm: riskAssessment.risk_level,
        signalQuality: ppgAnalysis.session_analysis.valid_hr_measurements > 10 ? 'Good' : 'Fair',
        heartRateVariability: heartRateMetrics.std_deviation,
        sessionDuration: ppgAnalysis.session_analysis.session_duration,
        riskLevel: riskAssessment.risk_level,
        riskPercentage: riskAssessment.risk_percentage,
        recommendations: ppgAnalysis.recommendations,
        userId: user.id,
        timestamp: new Date(),
        dataSource: 'PPG',
        measurementType: 'real-time'
      });
      
      // Reload measurements to show the new data
      await loadRecentMeasurements();
      
      console.log('✅ PPG measurement saved successfully!');
      
    } catch (error) {
      console.error('Error saving PPG measurement:', error);
      console.error('❌ Failed to save measurement. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle manual data entry and save to Firebase
  const handleManualDataSave = async (heartData) => {
    try {
      setLoading(true);
      
      await addDoc(collection(db, 'measurements'), {
        bpm: heartData.bpm,
        rhythm: heartData.rhythm || 'Normal',
        bloodPressure: heartData.bloodPressure,
        stress: heartData.stress,
        activity: heartData.activity,
        notes: heartData.notes,
        userId: user.id,
        timestamp: new Date(),
        dataSource: 'Manual',
        measurementType: 'manual-entry'
      });
      
      await loadRecentMeasurements();
      console.log('✅ Manual data saved successfully!');
      
    } catch (error) {
      console.error('Error saving manual data:', error);
      console.error('❌ Failed to save data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Render different views based on current selection
  const renderCurrentView = () => {
    switch (currentView) {
      case 'ppg':
        return (
          <PPGMonitor 
            onMeasurementComplete={handlePPGMeasurementComplete}
            user={user}
          />
        );
      case 'stats':
        return (
          <div className="stats-view">
            <h2>📈 Heart Health Dashboard</h2>
            {loading ? (
              <p>Loading measurements...</p>
            ) : (
              <div className="measurements-grid">
                {recentMeasurements.map((measurement) => (
                  <HeartStatsCard 
                    key={measurement.id}
                    measurement={measurement}
                  />
                ))}
                {recentMeasurements.length === 0 && (
                  <p>No measurements yet. Start with a PPG scan!</p>
                )}
              </div>
            )}
          </div>
        );
      case 'ai':
        return (
          <AIAssistant 
            recentMeasurements={recentMeasurements}
            onDataSave={handleManualDataSave}
          />
        );
      default:
        return (
          <div className="home-view">
            <div className="hero-section">
              <h1>💖 PulseHer</h1>
              <p>Your AI-powered heart health companion with real-time PPG monitoring</p>
            </div>
            
            <div className="quick-actions">
              <button 
                className="action-btn primary"
                onClick={() => setCurrentView('ppg')}
              >
                🩸 Start PPG Heart Rate Scan
              </button>
              
              <button 
                className="action-btn secondary"
                onClick={() => setCurrentView('stats')}
              >
                📊 View Health Dashboard
              </button>
              
              <button 
                className="action-btn secondary"
                onClick={() => setCurrentView('ai')}
              >
                🤖 AI Health Assistant
              </button>
            </div>
            
            {recentMeasurements.length > 0 && (
              <div className="recent-summary">
                <h3>Latest Reading</h3>
                <div className="latest-card">
                  <div className="metric">
                    <span className="value">{Math.round(recentMeasurements[0].bpm)}</span>
                    <span className="unit">BPM</span>
                  </div>
                  <div className="status">
                    <span className={`risk-level ${recentMeasurements[0].riskLevel?.toLowerCase().replace(' ', '-')}`}>
                      {recentMeasurements[0].riskLevel || recentMeasurements[0].rhythm}
                    </span>
                  </div>
                  <div className="timestamp">
                    {recentMeasurements[0].timestamp?.seconds 
                      ? new Date(recentMeasurements[0].timestamp.seconds * 1000).toLocaleString()
                      : new Date().toLocaleString()
                    }
                  </div>
                </div>
              </div>
            )}
            
            <div className="features-preview">
              <h3>🚀 Features</h3>
              <div className="features-grid">
                <div className="feature-card">
                  <h4>🩸 PPG Heart Monitoring</h4>
                  <p>Real-time heart rate detection using your camera</p>
                </div>
                <div className="feature-card">
                  <h4>🤖 AI Risk Assessment</h4>
                  <p>Machine learning powered cardiovascular risk analysis</p>
                </div>
                <div className="feature-card">
                  <h4>💾 Cloud Storage</h4>
                  <p>Secure Firebase data storage for your health records</p>
                </div>
                <div className="feature-card">
                  <h4>📊 Health Analytics</h4>
                  <p>Comprehensive tracking and trend analysis</p>
                </div>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="pulseher-app">
      {/* Navigation Header */}
      <nav className="app-nav">
        <div className="nav-brand">
          <h1>🫀 CardiIQ</h1>
        </div>
        <div className="nav-links">
          <button 
            className={currentView === 'home' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setCurrentView('home')}
          >
            🏠 Home
          </button>
          <button 
            className={currentView === 'ppg' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setCurrentView('ppg')}
          >
            🩸 PPG Scan
          </button>
          <button 
            className={currentView === 'stats' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setCurrentView('stats')}
          >
            📊 Dashboard
          </button>
          <button 
            className={currentView === 'ai' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setCurrentView('ai')}
          >
            🤖 AI Assistant
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="app-main">
        {renderCurrentView()}
      </main>
      
      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Saving to Firebase...</p>
        </div>
      )}
    </div>
  );
}

export default App;