import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import avatarImg from '../assets/fino-bg.png';

const ChatContainer = ({ messages, isLoading }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="chat-container">
      {messages.map((msg) => (
        <div key={msg.id} className={`message-wrapper ${msg.role}`}>
          <div className="message-meta">
            {msg.role === 'ai' && (
              <div className="ai-avatar" style={{ background: 'transparent' }}>
                <img src={avatarImg} alt="AI" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
              </div>
            )}
            <span>{msg.role === 'ai' ? 'Fino AI Assistant' : 'You'}</span>
            <span> • </span>
            <span>{msg.timestamp}</span>
          </div>
          <div className="message-bubble">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="message-wrapper ai">
          <div className="message-meta">
            <div className="ai-avatar" style={{ background: 'transparent' }}>
              <img src={avatarImg} alt="AI" style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '4px' }} />
            </div>
            <span>Fino AI Assistant is analyzing...</span>
          </div>
          <div className="message-bubble" style={{ display: 'flex', padding: '0.8rem 1.2rem' }}>
            <div className="typing-dots">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
};

export default ChatContainer;