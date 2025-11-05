import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <Link to="/" className="brand-link">
          <h1>💖 PulseHer</h1>
        </Link>
      </div>
      <div className="nav-links">
        <Link 
          to="/" 
          className={location.pathname === '/' ? 'nav-link active' : 'nav-link'}
        >
          🏠 Home
        </Link>
        <Link 
          to="/scan" 
          className={location.pathname === '/scan' ? 'nav-link active' : 'nav-link'}
        >
          🩸 Scan
        </Link>
        <Link 
          to="/insights" 
          className={location.pathname === '/insights' ? 'nav-link active' : 'nav-link'}
        >
          📊 Insights
        </Link>
        <Link 
          to="/learn" 
          className={location.pathname === '/learn' ? 'nav-link active' : 'nav-link'}
        >
          📚 Learn
        </Link>
        <Link 
          to="/profile" 
          className={location.pathname === '/profile' ? 'nav-link active' : 'nav-link'}
        >
          👤 Profile
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;