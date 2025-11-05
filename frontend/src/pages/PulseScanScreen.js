import React from 'react';
import PPGMonitor from '../components/PPGMonitor';

const PulseScanScreen = () => {
  return (
    <div className="pulse-scan-screen">
      <div style={{ 
        minHeight: 'calc(100vh - 80px)',
        background: 'linear-gradient(135deg, #ffeef8 0%, #f8e6f0 50%, #ffd6f0 100%)',
        padding: '2rem'
      }}>
        <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
          <h1 style={{ color: '#e91e63', marginBottom: '2rem' }}>🩸 PPG Heart Rate Scanner</h1>
          <PPGMonitor />
        </div>
      </div>
    </div>
  );
};

export default PulseScanScreen;