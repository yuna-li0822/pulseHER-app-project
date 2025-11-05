// 💗 PulseHER Master Blueprint (v10.0) - React Version
// Purpose: Research-grade clinical women's health platform with cycle-aware PPG analytics
// Complete system: camera PPG → HR/HRV extraction → adaptive indices (ABI,CVR,CSI) → explainable flags
// Privacy-first, research-ready with clinical validation and export capabilities

import React, { useState } from 'react';
import './App.css';
import CameraPPG from './components/CameraPPG';

function App() {
  const [currentScreen, setCurrentScreen] = useState('welcome');
  const [menuOpen, setMenuOpen] = useState(false);

  const showScreen = (screenId) => {
    setCurrentScreen(screenId);
    setMenuOpen(false);
    // After the screen updates, scroll the recommendations element into view with an offset
    // so it sits directly under the sticky header. This targets the React app path.
    setTimeout(() => {
      try {
        if (screenId === 'recommendations') {
          const el = document.getElementById('recommendations');
          const header = document.querySelector('.header');
          const headerH = header ? header.offsetHeight : 0;
          if (el) {
            const top = el.getBoundingClientRect().top + window.scrollY - headerH - 8; // small buffer
            window.scrollTo({ top, left: 0, behavior: 'auto' });
          } else {
            // fallback: scroll main to top
            const main = document.querySelector('.main-content');
            if (main) main.scrollTop = 0;
            window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
          }
        } else {
          // default: scroll to top for other screens
          const main = document.querySelector('.main-content');
          if (main) main.scrollTop = 0;
          window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
        }
      } catch (e) { /* ignore scroll errors */ }
    }, 60);
  };

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo">💖 PulseHER</div>
        <button className="menu-toggle" onClick={toggleMenu}>☰</button>
      </header>
      
      {/* Navigation Menu */}
      <nav className={`nav-menu ${menuOpen ? 'open' : ''}`}>
        <button className="nav-close" onClick={toggleMenu}>✕</button>
        <div className="nav-links">
          <a href="#" className={`nav-link ${currentScreen === 'welcome' ? 'active' : ''}`} onClick={() => showScreen('welcome')}>🏠 Home</a>
          <a href="#" className={`nav-link ${currentScreen === 'onboarding' ? 'active' : ''}`} onClick={() => showScreen('onboarding')}>👋 Get Started</a>
          <a href="#" className={`nav-link ${currentScreen === 'dashboard' ? 'active' : ''}`} onClick={() => showScreen('dashboard')}>📊 Dashboard</a>
          <a href="#" className={`nav-link ${currentScreen === 'ppg' ? 'active' : ''}`} onClick={() => showScreen('ppg')}>💓 PPG Recording</a>
          <a href="#" className={`nav-link ${currentScreen === 'results' ? 'active' : ''}`} onClick={() => showScreen('results')}>📊 Results</a>
          <a href="#" className={`nav-link ${currentScreen === 'insights' ? 'active' : ''}`} onClick={() => showScreen('insights')}>🧠 AI Insights</a>
          <a href="#" className={`nav-link ${currentScreen === 'cycle-analytics' ? 'active' : ''}`} onClick={() => showScreen('cycle-analytics')}>📈 Cycle Analytics</a>
          <a href="#" className={`nav-link ${currentScreen === 'recommendations' ? 'active' : ''}`} onClick={() => showScreen('recommendations')}>💡 Recommendations</a>
        </div>
      </nav>
      
      {/* Main Content */}
      <main className="main-content">
        {/* Welcome Screen */}
        {currentScreen === 'welcome' && (
          <div className="screen active welcome-screen">
            <h1 className="welcome-title">PulseHER</h1>
            <p className="welcome-subtitle">🌸 Your Heart Health, Your Way 🌸<br />Empowering women with AI-driven insights for cardiovascular wellness</p>
            
            <div className="heart-animation">💖</div>
            
            <button className="cta-button" onClick={() => showScreen('onboarding')}>
              🚀 Start Your Journey
            </button>
            <button className="cta-button" onClick={() => showScreen('dashboard')}>
              📊 View Dashboard
            </button>
            
            <div className="card">
              <h3 className="pink-accent">🌟 Features</h3>
              <ul style={{ textAlign: 'left', marginTop: '15px' }}>
                <li style={{ margin: '10px 0', color: '#2d1b2e' }}>💕 Real-time heart rate monitoring</li>
                <li style={{ margin: '10px 0', color: '#2d1b2e' }}>🔄 Menstrual cycle integration</li>
                <li style={{ margin: '10px 0', color: '#2d1b2e' }}>🤖 AI-powered health insights</li>
                <li style={{ margin: '10px 0', color: '#2d1b2e' }}>📱 Personalized recommendations</li>
              </ul>
            </div>
          </div>
        )}

        {/* Dashboard Screen */}
        {currentScreen === 'dashboard' && (
          <div className="screen active">
            <div className="dashboard-header">
              <h2 className="dashboard-title">📊 Clinical Dashboard v10.0</h2>
              <p style={{ color: '#2d1b2e', fontSize: '0.9rem' }}>Cycle-aware analytics with adaptive indices & explainable flags</p>
            </div>
            
            <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
              <div className="stat-card">
                <div className="stat-value" style={{ fontSize: '1.8rem' }}>72</div>
                <div className="stat-label">💓 Mean HR</div>
                <div style={{ fontSize: '0.7rem', color: '#c2185b' }}>Phase-adjusted</div>
              </div>
              <div className="stat-card">
                <div className="stat-value" style={{ fontSize: '1.8rem' }}>45</div>
                <div className="stat-label">🌊 RMSSD</div>
                <div style={{ fontSize: '0.7rem', color: '#c2185b' }}>ms</div>
              </div>
              <div className="stat-card">
                <div className="stat-value" style={{ fontSize: '1.8rem' }}>85</div>
                <div className="stat-label">😴 Sleep</div>
                <div style={{ fontSize: '0.7rem', color: '#9c27b0' }}>/100</div>
              </div>
              <div className="stat-card">
                <div className="stat-value" style={{ fontSize: '1.8rem' }}>92</div>
                <div className="stat-label">🏃‍♀️ Activity</div>
                <div style={{ fontSize: '0.7rem', color: '#4caf50' }}>/100</div>
              </div>
            </div>

            <div className="card">
              <h3 className="pink-accent">🌙 Cycle Ring Visualization</h3>
              <div style={{ display: 'flex', justifyContent: 'center', margin: '15px 0' }}>
                <div style={{ position: 'relative', width: '220px', height: '220px' }}>
                  <svg width="220" height="220" style={{ position: 'absolute', top: 0, left: 0 }}>
                    <circle cx="110" cy="110" r="90" fill="none" stroke="rgba(248,187,217,0.2)" strokeWidth="20"/>
                    <circle cx="110" cy="110" r="90" fill="none" stroke="#E91E63" strokeWidth="20" 
                            strokeDasharray="102 468" strokeDashoffset="0" opacity="0.8"/>
                    <circle cx="110" cy="110" r="90" fill="none" stroke="#FF9800" strokeWidth="20"
                            strokeDasharray="162 408" strokeDashoffset="-102" opacity="0.8"/>
                    <circle cx="110" cy="110" r="90" fill="none" stroke="#FFD54F" strokeWidth="20"
                            strokeDasharray="40 530" strokeDashoffset="-264" opacity="0.9"/>
                    <circle cx="110" cy="110" r="90" fill="none" stroke="#9C27B0" strokeWidth="20"
                            strokeDasharray="264 306" strokeDashoffset="-304" opacity="0.8"/>
                    <circle cx="200" cy="110" r="8" fill="#c2185b" stroke="white" strokeWidth="3">
                      <animate attributeName="r" values="8;12;8" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <text x="110" y="105" textAnchor="middle" fontSize="16" fill="#c2185b" fontWeight="bold">Day 14</text>
                    <text x="110" y="120" textAnchor="middle" fontSize="12" fill="#4a2c4a">Ovulation</text>
                    <text x="110" y="135" textAnchor="middle" fontSize="10" fill="#666">Peak Fertility</text>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* PPG Screen */}
        {currentScreen === 'ppg' && (
          <div className="screen active">
            <div className="card">
              <h2 className="pink-accent">📱 Clinical-Grade PPG Analytics 📱</h2>
              <p style={{ color: '#4a2c4a', textAlign: 'center', margin: '15px 0', fontSize: '0.9rem' }}>
                Research-quality PPG capture with cycle-aware HRV analysis<br/>
                <span style={{ fontSize: '0.8rem', opacity: 0.8 }}>Extracting HR, RMSSD, SDNN, LF/HF → ABI/CVR/CSI indices</span>
              </p>

              {/* Render the CameraPPG component which handles camera access and live waveform */}
              <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <CameraPPG />
              </div>
            </div>
          </div>
        )}

        {/* Other screens with placeholder content */}
        {currentScreen === 'insights' && (
          <div className="screen active">
            <div className="card">
              <h2 className="pink-accent">🤖 AI Health Insights 🤖</h2>
              <p style={{ color: '#2d1b2e', margin: '20px 0' }}>💖 Your heart health analysis will appear here...</p>
            </div>
          </div>
        )}

        {/* Recommendations / Tips Screen */}
        {currentScreen === 'recommendations' && (
          <div id="recommendations" className="screen active">
            <div className="card">
              <h2 className="pink-accent">💡 Personalized Recommendations 💡</h2>
              <h3 className="pink-accent">🏃‍♀️ Exercise Recommendations</h3>
              <ul style={{ margin: '10px 0', color: '#2d1b2e' }}>
                <li style={{ margin: '6px 0' }}>💪 Monday: 30-min moderate cardio</li>
                <li style={{ margin: '6px 0' }}>🧘‍♀️ Tuesday: Gentle yoga and stretching</li>
                <li style={{ margin: '6px 0' }}>🏋️‍♀️ Wednesday: Strength training</li>
                <li style={{ margin: '6px 0' }}>🚶‍♀️ Thursday: Recovery walk</li>
              </ul>

              <h3 className="pink-accent">🥗 Nutrition for Heart Health</h3>
              <ul style={{ margin: '10px 0', color: '#2d1b2e' }}>
                <li style={{ margin: '6px 0' }}>🫐 Berries for antioxidants</li>
                <li style={{ margin: '6px 0' }}>🥑 Healthy fats like avocado and nuts</li>
                <li style={{ margin: '6px 0' }}>🐟 Include omega-3 rich fish</li>
              </ul>

              <h3 className="pink-accent">😴 Sleep & Recovery</h3>
              <p style={{ color: '#2d1b2e', margin: '8px 0' }}>Aim for 7-9 hours of quality sleep — limit screens 1 hour before bed.</p>
            </div>
          </div>
        )}

        {/* Add other screens as needed */}
        {currentScreen !== 'welcome' && currentScreen !== 'dashboard' && currentScreen !== 'ppg' && currentScreen !== 'insights' && (
          <div className="screen active">
            <div className="card">
              <h2 className="pink-accent">Coming Soon...</h2>
              <p style={{ color: '#2d1b2e' }}>This feature is under development</p>
            </div>
          </div>
        )}
      </main>
      
      {/* Bottom Navigation */}
      <nav className="bottom-nav">
        <a href="#" className={`nav-item ${currentScreen === 'welcome' ? 'active' : ''}`} onClick={() => showScreen('welcome')}>
          <div className="nav-icon">🏠</div>
          <div className="nav-label">Home</div>
        </a>
        <a href="#" className={`nav-item ${currentScreen === 'dashboard' ? 'active' : ''}`} onClick={() => showScreen('dashboard')}>
          <div className="nav-icon">📊</div>
          <div className="nav-label">Dashboard</div>
        </a>
        <a href="#" className={`nav-item ${currentScreen === 'ppg' ? 'active' : ''}`} onClick={() => showScreen('ppg')}>
          <div className="nav-icon">💓</div>
          <div className="nav-label">PPG</div>
        </a>
        <a href="#" className={`nav-item ${currentScreen === 'insights' ? 'active' : ''}`} onClick={() => showScreen('insights')}>
          <div className="nav-icon">🧠</div>
          <div className="nav-label">Insights</div>
        </a>
        <a href="#" className={`nav-item ${currentScreen === 'recommendations' ? 'active' : ''}`} onClick={() => showScreen('recommendations')}>
          <div className="nav-icon">💡</div>
          <div className="nav-label">Tips</div>
        </a>
      </nav>
    </div>
  );
}

export default App;
