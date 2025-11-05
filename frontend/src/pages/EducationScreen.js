import React from 'react';

const EducationScreen = () => {
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
        <h1>📚 Heart Health Education</h1>
        <p>Educational content and tips coming soon...</p>
      </div>
    </div>
  );
};

export default EducationScreen;