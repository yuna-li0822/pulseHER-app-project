import React from 'react';
import './SaveSessionModal.css';

const SaveSessionModal = ({ onSave, onDelete, onClose }) => {
  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>Session Complete</h2>
        <p>Would you like to save this session to your longitudinal tracking?</p>
        <div className="modal-actions">
          <button className="save-btn" onClick={onSave}>Save</button>
          <button className="delete-btn" onClick={onDelete}>Delete</button>
        </div>
        <button className="close-btn" onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export default SaveSessionModal;
