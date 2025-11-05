import React, { useState, useEffect } from 'react';
import './HomeScreen.css';

const HomeScreen = () => {
  const [recentData, setRecentData] = useState({
    heartRate: 72,
    lastScan: '2 hours ago',
    weeklyAvg: 68,
    status: 'Normal'
  });

  return (
    <div className="home-screen">
      <div className="home-container">
        <div className="welcome-section">
          <h1>🏠 Your Heart Health Dashboard</h1>
          <p>Welcome back! Here's your latest heart health summary.</p>
        </div>

        <div className="quick-stats">
          <div className="stat-card">
            <div className="stat-icon">💓</div>
            <div className="stat-info">
              <h3>{recentData.heartRate} BPM</h3>
              <p>Current Heart Rate</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">📊</div>
            <div className="stat-info">
              <h3>{recentData.weeklyAvg} BPM</h3>
              <p>Weekly Average</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">✅</div>
            <div className="stat-info">
              <h3>{recentData.status}</h3>
              <p>Health Status</p>
            </div>
          </div>
        </div>

        <div className="action-cards">
          <div className="action-card">
            <h3>🩸 Quick Scan</h3>
            <p>Take a new heart rate measurement</p>
            <button className="action-button">Start Scan</button>
          </div>
          <div className="action-card">
            <h3>📈 View Trends</h3>
            <p>See your heart health over time</p>
            <button className="action-button">View Insights</button>
          </div>
          <div className="action-card">
            <h3>📚 Health Tips</h3>
            <p>Learn about heart health</p>
            <button className="action-button">Learn More</button>
          </div>
        </div>

        <div className="recent-activity">
          <h2>Recent Activity</h2>
          <div className="activity-item">
            <span className="activity-icon">🩸</span>
            <div className="activity-info">
              <p>Heart rate scan completed</p>
              <span className="activity-time">{recentData.lastScan}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomeScreen;