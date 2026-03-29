import React from 'react';

const Header = ({ isHealthy }) => {
  return (
    <header className="header">
      <div className="header-title">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Fino Smart Search
      </div>
      <div className="status-indicator">
        <div className={`status-dot ${isHealthy ? 'online' : ''}`}></div>
        {isHealthy ? 'Online' : 'Offline'}
      </div>
    </header>
  );
};

export default Header;
