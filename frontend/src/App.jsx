import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import i18n from './i18n';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ChatContainer from './components/ChatContainer';
import MessageInput from './components/MessageInput';
import LanguageSplash from './components/LanguageSplash';

const API_BASE_URL = 'https://shrut04-fino-backend-api.hf.space';

function App() {
  // ── Splash / Language state ────────────────────────────────────────────────
  const [showSplash, setShowSplash] = useState(() => {
    // Only show on first visit per session
    return !sessionStorage.getItem('fino_language');
  });
  const [language, setLanguage] = useState(() => {
    const saved = sessionStorage.getItem('fino_language');
    return saved ? JSON.parse(saved) : { code: 'en', name: 'English', native: 'English' };
  });

  // ── Chat state ─────────────────────────────────────────────────────────────
  const [messages, setMessages] = useState(() => {
    const saved = sessionStorage.getItem('fino_chat_history');
    return saved ? JSON.parse(saved) : [];
  });
  const [inputValue, setInputValue] = useState('');
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

  const { t } = useTranslation();

  const handleLanguageSelect = (lang) => {
    setLanguage(lang);
    sessionStorage.setItem('fino_language', JSON.stringify(lang));
    i18n.changeLanguage(lang.code);
    document.documentElement.setAttribute('dir', lang.code === 'ur' ? 'rtl' : 'ltr');
    setShowSplash(false);
  };

  // Sync i18n language on mount if saved
  useEffect(() => {
    if (language && language.code) {
      i18n.changeLanguage(language.code);
      document.documentElement.setAttribute('dir', language.code === 'ur' ? 'rtl' : 'ltr');
    }
  }, []);

  useEffect(() => {
    sessionStorage.setItem('fino_chat_history', JSON.stringify(messages));
    if (messages.length > 0) setHasStarted(true);
  }, [messages]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

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
        body: JSON.stringify({ query, thread_id: threadId, language: language.code })
      });

      if (!response.ok) throw new Error('Backend error');

      const data = await response.json();
      let displayContent = data.answer;

      if (typeof displayContent === 'string' && displayContent.trim().startsWith('{')) {
        try {
          const parsed = JSON.parse(displayContent);
          displayContent = parsed.final_answer || displayContent;
        } catch (e) {
          console.error('Failed to parse AI JSON response', e);
        }
      }

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'ai',
        content: displayContent,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'ai',
        content: 'Identity connection lost. Please ensure the Fino intelligence core is operational.',
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
    <>
      {/* Language splash renders on top of everything */}
      {showSplash && (
        <LanguageSplash onLanguageSelect={handleLanguageSelect} />
      )}

      <div className="app-container">
        <Navbar
          healthStatus={healthStatus}
          isDarkMode={isDarkMode}
          toggleDarkMode={() => setIsDarkMode(!isDarkMode)}
          language={language}
          onChangeLanguage={() => setShowSplash(true)}
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
          language={language.name}
          setLanguage={(name) => setLanguage(prev => ({ ...prev, name }))}
        />
      </div>
    </>
  );
}

export default App;
