import React from 'react';
import logo from '../assets/fino-logo.png';

const Navbar = ({ healthStatus, isDarkMode, toggleDarkMode }) => {
  const styles = {
    navbar: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '10px 24px',
      background: 'var(--glass-bg)',
      backdropFilter: 'var(--glass-blur)',
      borderBottom: '1px solid var(--border-subtle)',
    },
    navBrand: {
      display: 'flex',
      alignItems: 'center',
    },
    logoImg: {
      height: '45px',
      width: 'auto',
      objectFit: 'contain',
      borderRadius: '8px',
    },
    navStatus: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      color: 'var(--text-secondary)',
      fontSize: '14px',
    },
    statusDot: (status) => ({
      width: '8px',
      height: '8px',
      borderRadius: '50%',
      backgroundColor:
        status === 'online'
          ? 'var(--success-green)'
          : status === 'degraded'
            ? '#f59e0b'
            : 'var(--error-red)',
    }),
    toggleBtn: {
      background: 'transparent',
      border: '1px solid var(--border-subtle)',
      color: 'var(--text-primary)',
      borderRadius: '6px',
      padding: '6px 12px',
      cursor: 'pointer',
      marginLeft: '16px',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all 0.3s ease',
    }
  };

  return (
    <nav style={styles.navbar}>
      <div style={styles.navBrand}>
        <img src={logo} alt="Fino Logo" style={styles.logoImg} />
      </div>

      <div style={styles.navStatus}>
        <div style={styles.statusDot(healthStatus)}></div>
        <span>
          {healthStatus === 'online'
            ? 'Online'
            : healthStatus === 'degraded'
              ? 'Degraded'
              : 'Offline'}
        </span>
        <button style={styles.toggleBtn} onClick={toggleDarkMode}>
          {isDarkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>
    </nav>
  );
};

export default Navbar;