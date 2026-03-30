import React, { useRef, useEffect } from 'react';

const MessageInput = ({
  inputValue,
  setInputValue,
  onSend,
  onClear,
  isLoading,
}) => {
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [inputValue]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="input-section">
      <div className="input-wrapper">
        <div className="input-container">
          <textarea
            ref={textareaRef}
            className="chat-input"
            placeholder="Ask a question..."
            rows="1"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
          ></textarea>

          <div className="action-buttons">
            <button
              className="icon-btn"
              onClick={onClear}
              title="Clear Chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 5H9l-7 7 7 7h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Z" /><line x1="18" y1="9" x2="12" y2="15" /><line x1="12" y1="9" x2="18" y2="15" /></svg>
            </button>
            <button
              className="send-btn"
              onClick={onSend}
              disabled={!inputValue.trim() || isLoading}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>
            </button>
          </div>
        </div>
      </div>

      <div style={{ fontSize: '1.0rem', color: '#6b7280', textAlign: 'center', marginTop: '8px', fontWeight: '400' }}>
        AI-generated responses can have errors. Please verify important information.
      </div>
    </div>
  );
};

export default MessageInput;
