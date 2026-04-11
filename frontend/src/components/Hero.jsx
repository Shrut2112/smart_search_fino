import React from 'react';
import { useTranslation } from 'react-i18next';

const Hero = ({ onSuggestionClick }) => {
  const { t, i18n } = useTranslation();
  
  const suggestions = t('hero.suggestions', { returnObjects: true }) || [
    "What is our refund policy?",
    "Show Q3 revenue data",
    "Summarize loan structure"
  ];

  return (
    <section className="hero-section" data-lang={i18n.language}>
      <h1 className="hero-title">{t('hero.title')}</h1>
      <p className="hero-subtitle">
        {t('hero.subtitle')}
      </p>
      <div className="suggestions-grid">
        {suggestions.map((text, idx) => (
          <button
            key={idx}
            className="suggestion-chip"
            onClick={() => onSuggestionClick(text)}
          >
            {text}
          </button>
        ))}
      </div>
    </section>
  );
};

export default Hero;
