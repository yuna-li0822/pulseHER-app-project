import React from 'react';

const InsightScreen = () => {
  return (
    <div style={{ 
      minHeight: 'calc(100vh - 80px)',
      background: 'linear-gradient(135deg, #ffeef8 0%, #f8e6f0 50%, #ffd6f0 100%)',
      padding: '2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <div style={{ textAlign: 'center', color: '#e91e63' }}>
        <h1>📊 Health Insights</h1>
        <p>Your personalized health analytics coming soon...</p>
      </div>
    </div>
  );
};

export default InsightScreen;