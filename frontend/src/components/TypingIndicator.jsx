import React from 'react';

const TypingIndicator = () => {
  return (
    <div className="message-wrapper ai">
      <div className="message-bubble ai typing-indicator">
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
      </div>
    </div>
  );
};

export default TypingIndicator;
