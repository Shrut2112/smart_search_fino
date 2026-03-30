import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ChatContainer from './components/ChatContainer';
import MessageInput from './components/MessageInput';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState(() => {
    const saved = sessionStorage.getItem('fino_chat_history');
    return saved ? JSON.parse(saved) : [];
  });
  const [inputValue, setInputValue] = useState('');
  const [language, setLanguage] = useState('English');
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState('offline');
  const [hasStarted, setHasStarted] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  const [threadId] = useState(() => {
    let id = sessionStorage.getItem('fino_thread_id');
    if (!id) {
      id = `thread_${Math.random().toString(36).substring(7)}`;
      sessionStorage.setItem('fino_thread_id', id);
    }
    return id;
  });

  useEffect(() => {
    sessionStorage.setItem('fino_chat_history', JSON.stringify(messages));
    if (messages.length > 0) setHasStarted(true);
  }, [messages]);


  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  // Health check polling
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        setHealthStatus(data.status === 'healthy' ? 'online' : 'degraded');
      } catch (error) {
        setHealthStatus('offline');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSend = async (overrideText) => {
    const query = overrideText || inputValue;
    if (!query.trim() || isLoading) return;

    if (!hasStarted) setHasStarted(true);

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg = { id: Date.now(), role: 'user', content: query, timestamp };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          thread_id: threadId
        })
      });

      if (!response.ok) throw new Error('Backend error');

      const data = await response.json();
      let displayContent = data.answer;

      // If the answer looks like JSON, try to extract the final_answer field
      if (typeof displayContent === 'string' && displayContent.trim().startsWith('{')) {
        try {
          const parsed = JSON.parse(displayContent);
          displayContent = parsed.final_answer || displayContent;
        } catch (e) {
          console.error("Failed to parse AI JSON response", e);
        }
      }

      const aiMsg = {
        id: Date.now() + 1,
        role: 'ai',
        content: displayContent, // Use the cleaned content
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'ai',
        content: "Identity connection lost. Please ensure the Fino intelligence core is operational.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setHasStarted(false);
    setInputValue('');
  };

  return (
    <div className="app-container">
      <Navbar
        healthStatus={healthStatus}
        isDarkMode={isDarkMode}
        toggleDarkMode={() => setIsDarkMode(!isDarkMode)}
      />

      <main className="main-content">
        {!hasStarted ? (
          <Hero onSuggestionClick={(text) => handleSend(text)} />
        ) : (
          <ChatContainer messages={messages} isLoading={isLoading} />
        )}
      </main>

      <MessageInput
        inputValue={inputValue}
        setInputValue={setInputValue}
        onSend={() => handleSend()}
        onClear={handleClear}
        isLoading={isLoading}
        language={language}
        setLanguage={setLanguage}
      />
    </div>
  );
}

export default App;
