import React from 'react';
import { useTranslation } from 'react-i18next';
import logo from '../assets/fino-logo.png';

const Navbar = ({ healthStatus, isDarkMode, toggleDarkMode, language, onChangeLanguage }) => {
  const { t } = useTranslation();
  const styles = {
    navbar: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '10px 24px',
      background: 'var(--glass-bg)',
      backdropFilter: 'var(--glass-blur)',
      borderBottom: '1px solid var(--border-subtle)',
      zIndex: 100,
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
    navRight: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
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
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all 0.3s ease',
    },
    langBtn: {
      display: 'flex',
      alignItems: 'center',
      gap: '7px',
      background: 'var(--accent-gold-transparent)',
      border: '1px solid var(--border-gold)',
      color: 'var(--accent-gold)',
      borderRadius: '8px',
      padding: '6px 13px',
      cursor: 'pointer',
      fontSize: '13px',
      fontWeight: '600',
      transition: 'all 0.25s ease',
      fontFamily: 'var(--font-body)',
    },
  };

  return (
    <nav style={styles.navbar}>
      <div style={styles.navBrand}>
        <img src={logo} alt="Fino Logo" style={styles.logoImg} />
      </div>

      <div style={styles.navRight}>
        {/* Language pill */}
        {language && (
          <button
            id="navbar-lang-btn"
            style={styles.langBtn}
            onClick={onChangeLanguage}
            title={t('nav.changeLanguage')}
          >
            <span style={{ fontSize: '16px' }}>🌐</span>
            <span>{language.native || language.name || 'English'}</span>
          </button>
        )}

        {/* Status indicator */}
        <div style={styles.navStatus}>
          <div style={styles.statusDot(healthStatus)} />
          <span>
            {healthStatus === 'online' ? t('nav.online') : healthStatus === 'degraded' ? t('nav.degraded') : t('nav.offline')}
          </span>
        </div>

        {/* Dark mode toggle */}
        <button style={styles.toggleBtn} onClick={toggleDarkMode}>
          {isDarkMode ? `☀️ ${t('nav.light')}` : `🌙 ${t('nav.dark')}`}
        </button>
      </div>
    </nav>
  );
};

export default Navbar;