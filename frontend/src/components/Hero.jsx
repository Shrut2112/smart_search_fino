import React from 'react';

const Hero = ({ onSuggestionClick }) => {
  const suggestions = [
    "What is our refund policy?",
    "Show Q3 revenue data",
    "Summarize loan structure"
  ];

  return (
    <section className="hero-section">
      <h1 className="hero-title">Welcome to Fino</h1>
      <p className="hero-subtitle">
        Your intelligent multilingual search assistant - ask anything about policies, structure, or data.
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
