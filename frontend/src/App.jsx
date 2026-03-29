import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ChatContainer from './components/ChatContainer';
import MessageInput from './components/MessageInput';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [language, setLanguage] = useState('English');
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState('offline');
  const [hasStarted, setHasStarted] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

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
        body: JSON.stringify({ query })
      });

      if (!response.ok) throw new Error('Backend error');

      const data = await response.json();
      const aiMsg = { 
        id: Date.now() + 1, 
        role: 'ai', 
        content: data.answer, 
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
