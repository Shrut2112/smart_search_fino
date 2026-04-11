import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

const languages = [
  { code: 'en',  name: 'English',    native: 'English',        script: 'A',   region: 'All India' },
  { code: 'hi',  name: 'Hindi',      native: 'हिंदी',           script: 'अ',   region: 'North India' },
  { code: 'mr',  name: 'Marathi',    native: 'मराठी',           script: 'म',   region: 'Maharashtra' },
  { code: 'gu',  name: 'Gujarati',   native: 'ગુજરાતી',         script: 'ગ',   region: 'Gujarat' },
  { code: 'bn',  name: 'Bengali',    native: 'বাংলা',           script: 'ব',   region: 'West Bengal' },
  { code: 'ta',  name: 'Tamil',      native: 'தமிழ்',           script: 'த',   region: 'Tamil Nadu' },
  { code: 'te',  name: 'Telugu',     native: 'తెలుగు',          script: 'త',   region: 'Andhra Pradesh' },
  { code: 'kn',  name: 'Kannada',    native: 'ಕನ್ನಡ',           script: 'ಕ',   region: 'Karnataka' },
  { code: 'ml',  name: 'Malayalam',  native: 'മലയാളം',          script: 'മ',   region: 'Kerala' },
  { code: 'pa',  name: 'Punjabi',    native: 'ਪੰਜਾਬੀ',          script: 'ਪ',   region: 'Punjab' },
  { code: 'or',  name: 'Odia',       native: 'ଓଡ଼ିଆ',           script: 'ଓ',   region: 'Odisha' },
  { code: 'ur',  name: 'Urdu',       native: 'اردو',            script: 'ا',   region: 'Uttar Pradesh' },
];

export default function LanguageSplash({ onLanguageSelect }) {
  const { t } = useTranslation();
  const [hoveredLang, setHoveredLang] = useState(null);
  const [selectedLang, setSelectedLang] = useState(null);

  const handleSelect = (lang) => {
    setSelectedLang(lang.code);
    setTimeout(() => {
      onLanguageSelect(lang);
    }, 500);
  };

  return (
    <div className={`splash-overlay ${selectedLang ? 'splash-exit' : ''}`}>
      {/* Ambient animated particles */}
      <div className="splash-particles">
        {[...Array(6)].map((_, i) => (
          <div key={i} className={`splash-particle splash-particle-${i + 1}`} />
        ))}
      </div>

      <div className="splash-content">
        {/* Logo + Header */}
        <div className="splash-header">
          <div className="splash-logo">
            <span>F</span>
          </div>
          <h1 className="splash-title">Fino Smart Search</h1>
          <p className="splash-subtitle">
            {t('splash.subtitle')}
            <br />
            <span className="splash-subtitle-native">अपनी भाषा चुनें • ਆਪਣੀ ਭਾਸ਼ਾ ਚੁਣੋ • आपली भाषा निवडा</span>
          </p>
        </div>

        {/* Language Grid */}
        <div className="lang-grid">
          {languages.map((lang) => (
            <button
              key={lang.code}
              id={`lang-btn-${lang.code}`}
              className={`lang-card ${selectedLang === lang.code ? 'lang-card-selected' : ''} ${hoveredLang === lang.code ? 'lang-card-hovered' : ''}`}
              onClick={() => handleSelect(lang)}
              onMouseEnter={() => setHoveredLang(lang.code)}
              onMouseLeave={() => setHoveredLang(null)}
            >
              <div className="lang-script-glyph">{lang.script}</div>
              <div className="lang-info">
                <span className="lang-native">{lang.native}</span>
                <span className="lang-english">{lang.name}</span>
                <span className="lang-region">{lang.region}</span>
              </div>
              {selectedLang === lang.code && (
                <div className="lang-check">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
              )}
            </button>
          ))}
        </div>

        {/* Footer */}
        <p className="splash-footer">
          {t('splash.footer')}
        </p>
      </div>
    </div>
  );
}
