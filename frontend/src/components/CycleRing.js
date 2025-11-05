import React, { useState, useEffect } from 'react';
import './CycleRing.css';

const CycleRing = ({ 
  cycleLength = 28, 
  currentDay = 1, 
  lastPeriodDate, 
  sessions = [],
  className = '' 
}) => {
  const [hoveredDay, setHoveredDay] = useState(null);
  const [selectedDay, setSelectedDay] = useState(null);

  // Calculate cycle phases
  const getPhaseInfo = (day) => {
    if (day >= 1 && day <= 5) {
      return { phase: 'menstrual', color: '#e74c3c', label: 'Menstrual' };
    } else if (day >= 6 && day <= Math.floor(cycleLength / 2) - 2) {
      return { phase: 'follicular', color: '#3498db', label: 'Follicular' };
    } else if (day >= Math.floor(cycleLength / 2) - 1 && day <= Math.floor(cycleLength / 2) + 1) {
      return { phase: 'ovulation', color: '#f39c12', label: 'Ovulation' };
    } else {
      return { phase: 'luteal', color: '#9b59b6', label: 'Luteal' };
    }
  };

  // Get session data for a specific day
  const getSessionForDay = (day) => {
    if (!lastPeriodDate || !sessions.length) return null;
    
    const targetDate = new Date(lastPeriodDate);
    targetDate.setDate(targetDate.getDate() + day - 1);
    
    return sessions.find(session => {
      const sessionDate = new Date(session.timestamp);
      return sessionDate.toDateString() === targetDate.toDateString();
    });
  };

  // Calculate ring positions
  const calculateRingPosition = (day, radius) => {
    // Start at top (12 o'clock) and go clockwise
    const angle = ((day - 1) / cycleLength) * 360 - 90; // -90 to start at top
    const radian = (angle * Math.PI) / 180;
    
    return {
      x: 150 + radius * Math.cos(radian), // 150 is center
      y: 150 + radius * Math.sin(radian)
    };
  };

  // Render day markers
  const renderDayMarkers = () => {
    const markers = [];
    const radius = 120;

    for (let day = 1; day <= cycleLength; day++) {
      const position = calculateRingPosition(day, radius);
      const phaseInfo = getPhaseInfo(day);
      const session = getSessionForDay(day);
      const isCurrentDay = day === currentDay;
      const isHovered = hoveredDay === day;
      const isSelected = selectedDay === day;

      // Determine marker size and style
      let markerClass = 'day-marker';
      if (isCurrentDay) markerClass += ' current-day';
      if (isHovered) markerClass += ' hovered';
      if (isSelected) markerClass += ' selected';
      if (session) markerClass += ' has-data';

      markers.push(
        <g key={day}>
          {/* Day marker circle */}
          <circle
            cx={position.x}
            cy={position.y}
            r={isCurrentDay ? 8 : session ? 6 : 4}
            fill={session ? phaseInfo.color : 'rgba(108, 117, 125, 0.3)'}
            stroke={isCurrentDay ? '#fff' : 'none'}
            strokeWidth={isCurrentDay ? 2 : 0}
            className={markerClass}
            onMouseEnter={() => setHoveredDay(day)}
            onMouseLeave={() => setHoveredDay(null)}
            onClick={() => setSelectedDay(day === selectedDay ? null : day)}
          />
          
          {/* Day number (show every 5th day or current day) */}
          {(day % 5 === 0 || isCurrentDay) && (
            <text
              x={position.x}
              y={position.y + 20}
              textAnchor="middle"
              fontSize="10"
              fill="#6c757d"
              className="day-label"
            >
              {day}
            </text>
          )}
          
          {/* HRV indicator for days with data */}
          {session && session.indices && (
            <circle
              cx={position.x}
              cy={position.y}
              r={12}
              fill="none"
              stroke={getHRVColor(session.indices.abi)}
              strokeWidth={2}
              opacity={0.7}
              className="hrv-indicator"
            />
          )}
        </g>
      );
    }

    return markers;
  };

  // Get HRV color based on ABI score
  const getHRVColor = (abi) => {
    if (abi > 60) return '#e74c3c'; // High sympathetic
    if (abi > 40) return '#f39c12'; // Balanced
    return '#27ae60'; // Parasympathetic
  };

  // Render phase segments
  const renderPhaseSegments = () => {
    const segments = [];
    const radius = 90;
    const strokeWidth = 20;

    let currentPhase = '';
    let segmentStart = 1;

    for (let day = 1; day <= cycleLength + 1; day++) {
      const phaseInfo = getPhaseInfo(day <= cycleLength ? day : 1);
      
      if (phaseInfo.phase !== currentPhase || day === cycleLength + 1) {
        if (currentPhase) {
          // Create arc for the phase segment
          const startAngle = ((segmentStart - 1) / cycleLength) * 360 - 90;
          const endAngle = ((day - 1 - 1) / cycleLength) * 360 - 90;
          
          const startRadian = (startAngle * Math.PI) / 180;
          const endRadian = (endAngle * Math.PI) / 180;
          
          const x1 = 150 + radius * Math.cos(startRadian);
          const y1 = 150 + radius * Math.sin(startRadian);
          const x2 = 150 + radius * Math.cos(endRadian);
          const y2 = 150 + radius * Math.sin(endRadian);
          
          const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;
          
          const pathData = [
            'M', x1, y1,
            'A', radius, radius, 0, largeArcFlag, 1, x2, y2
          ].join(' ');

          const prevPhaseInfo = getPhaseInfo(segmentStart);
          
          segments.push(
            <path
              key={`${currentPhase}-${segmentStart}`}
              d={pathData}
              fill="none"
              stroke={prevPhaseInfo.color}
              strokeWidth={strokeWidth}
              opacity={0.6}
              className="phase-segment"
            />
          );
        }
        
        currentPhase = phaseInfo.phase;
        segmentStart = day;
      }
    }

    return segments;
  };

  // Render phase labels
  const renderPhaseLabels = () => {
    const phases = [
      { phase: 'menstrual', day: 3, color: '#e74c3c' },
      { phase: 'follicular', day: Math.floor(cycleLength * 0.25), color: '#3498db' },
      { phase: 'ovulation', day: Math.floor(cycleLength * 0.5), color: '#f39c12' },
      { phase: 'luteal', day: Math.floor(cycleLength * 0.75), color: '#9b59b6' }
    ];

    return phases.map(({ phase, day, color }) => {
      const position = calculateRingPosition(day, 60);
      const label = phase.charAt(0).toUpperCase() + phase.slice(1);
      
      return (
        <text
          key={phase}
          x={position.x}
          y={position.y}
          textAnchor="middle"
          fontSize="11"
          fill={color}
          fontWeight="600"
          className="phase-label"
        >
          {label}
        </text>
      );
    });
  };

  // Current day info
  const currentPhaseInfo = getPhaseInfo(currentDay);
  const currentSession = getSessionForDay(currentDay);

  return (
    <div className={`cycle-ring-container ${className}`}>
      <div className="cycle-ring-header">
        <h3>Cycle Tracking</h3>
        <div className="current-phase">
          <span 
            className="phase-dot" 
            style={{ backgroundColor: currentPhaseInfo.color }}
          ></span>
          <span className="phase-name">{currentPhaseInfo.label} Phase</span>
          <span className="cycle-day">Day {currentDay}</span>
        </div>
      </div>

      <div className="ring-svg-container">
        <svg width="300" height="300" className="cycle-ring-svg">
          {/* Background circle */}
          <circle
            cx="150"
            cy="150"
            r="90"
            fill="none"
            stroke="rgba(108, 117, 125, 0.1)"
            strokeWidth="20"
          />
          
          {/* Phase segments */}
          {renderPhaseSegments()}
          
          {/* Center circle */}
          <circle
            cx="150"
            cy="150"
            r="40"
            fill="rgba(255, 255, 255, 0.9)"
            stroke="rgba(214, 51, 132, 0.2)"
            strokeWidth="1"
          />
          
          {/* Center text */}
          <text
            x="150"
            y="145"
            textAnchor="middle"
            fontSize="12"
            fill="#6c757d"
            fontWeight="600"
          >
            Day
          </text>
          <text
            x="150"
            y="160"
            textAnchor="middle"
            fontSize="20"
            fill="#d63384"
            fontWeight="700"
          >
            {currentDay}
          </text>
          
          {/* Day markers */}
          {renderDayMarkers()}
          
          {/* Phase labels */}
          {renderPhaseLabels()}
        </svg>
      </div>

      {/* Day detail popup */}
      {(hoveredDay || selectedDay) && (
        <div className="day-detail-popup">
          <DayDetail 
            day={hoveredDay || selectedDay}
            phaseInfo={getPhaseInfo(hoveredDay || selectedDay)}
            session={getSessionForDay(hoveredDay || selectedDay)}
            onClose={() => {
              setHoveredDay(null);
              setSelectedDay(null);
            }}
          />
        </div>
      )}

      {/* Legend */}
      <div className="cycle-legend">
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#e74c3c' }}></div>
          <span>Menstrual</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#3498db' }}></div>
          <span>Follicular</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#f39c12' }}></div>
          <span>Ovulation</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#9b59b6' }}></div>
          <span>Luteal</span>
        </div>
      </div>

      {/* Expected changes info */}
      <div className="expected-changes">
        <h4>Expected Changes in {currentPhaseInfo.label} Phase:</h4>
        <ExpectedChanges phase={currentPhaseInfo.phase} />
      </div>
    </div>
  );
};

// Day detail component
const DayDetail = ({ day, phaseInfo, session, onClose }) => {
  return (
    <div className="day-detail">
      <div className="detail-header">
        <h4>Day {day}</h4>
        <button onClick={onClose} className="close-button">×</button>
      </div>
      
      <div className="detail-content">
        <div className="phase-info">
          <span className="phase-name" style={{ color: phaseInfo.color }}>
            {phaseInfo.label} Phase
          </span>
        </div>
        
        {session ? (
          <div className="session-data">
            <div className="metric-row">
              <span>Heart Rate:</span>
              <span>{session.metrics?.hr?.toFixed(1) || '--'} BPM</span>
            </div>
            <div className="metric-row">
              <span>HRV (RMSSD):</span>
              <span>{session.metrics?.rmssd?.toFixed(1) || '--'} ms</span>
            </div>
            {session.indices && (
              <>
                <div className="metric-row">
                  <span>ABI Score:</span>
                  <span>{session.indices.abi?.toFixed(1) || '--'}</span>
                </div>
                <div className="metric-row">
                  <span>Stress Index:</span>
                  <span>{session.indices.csi?.toFixed(1) || '--'}</span>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="no-data">
            <p>No measurement for this day</p>
          </div>
        )}
      </div>
    </div>
  );
};

// Expected changes component
const ExpectedChanges = ({ phase }) => {
  const changes = {
    menstrual: [
      "Heart rate may be slightly elevated",
      "HRV typically lower due to inflammation",
      "Energy levels often reduced"
    ],
    follicular: [
      "Heart rate generally stable",
      "HRV gradually improving",
      "Energy levels increasing"
    ],
    ovulation: [
      "Heart rate may peak around ovulation",
      "HRV changes due to hormonal surge",
      "Peak fertility and energy"
    ],
    luteal: [
      "Heart rate elevation in late luteal phase",
      "HRV may decrease before next cycle",
      "Possible PMS symptoms affecting metrics"
    ]
  };

  return (
    <ul className="expected-changes-list">
      {changes[phase]?.map((change, index) => (
        <li key={index}>{change}</li>
      ))}
    </ul>
  );
};

export default CycleRing;