import React from 'react';
import { Link } from 'react-router-dom';
import './WelcomeScreen.css';

const WelcomeScreen = () => {
  return (
    <div className="welcome-screen">
      <div className="welcome-container">
        <div className="hero-section">
          <h1>💖 Welcome to PulseHer</h1>
          <p className="tagline">Your AI-powered heart health companion</p>
          <p className="description">
            Track your cardiovascular health with advanced PPG technology, 
            get personalized insights, and take control of your heart health journey.
          </p>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🩸</div>
            <h3>PPG Heart Monitoring</h3>
            <p>Real-time heart rate detection using your camera</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>AI Health Insights</h3>
            <p>Machine learning powered cardiovascular analysis</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Health Tracking</h3>
            <p>Comprehensive tracking and trend analysis</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🌸</div>
            <h3>Women-Focused</h3>
            <p>Designed specifically for women's heart health</p>
          </div>
        </div>

        <div className="cta-section">
          <Link to="/scan" className="cta-button">
            🩸 Start Heart Rate Scan
          </Link>
          <Link to="/home" className="cta-button secondary">
            📊 View Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;