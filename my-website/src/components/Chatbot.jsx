import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! I am your AI assistant. How can I help you with the Hackathon Book?', sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send to HuggingFace backend
      const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'https://jiyaMughal-hackathon-chatbot-backend.hf.space';
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,  // Changed from 'message' to 'query' to match backend
          context: null,
          selected_text: null
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();

      // Add bot response - Changed from 'reply' to 'response' to match backend
      const botMessage = {
        id: Date.now() + 1,
        text: data.response || 'Sorry, I could not understand that.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="chatbot-header-content">
              <Bot size={18} className="chatbot-header-icon" />
              <h3>AI Assistant</h3>
            </div>
            <button className="chatbot-close" onClick={toggleChat}>
              <X size={20} />
            </button>
          </div>
          <div className="chatbot-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.sender}-message`}
              >
                <div className="message-content">
                  <div className="message-icon">
                    {message.sender === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <span>{message.text}</span>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chatbot-input-area">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="chatbot-input"
              rows="1"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
              className="chatbot-send-button"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      ) : (
        <button className="chatbot-toggle" onClick={toggleChat}>
          <MessageCircle size={24} />
        </button>
      )}
    </div>
  );
};

export default Chatbot;