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

  const handleSend = () => {
    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    // Simulate bot response with helpful messages about the book
    setIsLoading(true);

    setTimeout(() => {
      const botResponses = [
        "I'm a frontend-only chatbot. For questions about the Physical AI & Humanoid Robotics book, please check the documentation sections.",
        "This is a frontend-only version of the chatbot. The full AI functionality requires backend services.",
        "To learn about Physical AI & Humanoid Robotics, explore the book sections in the navigation menu.",
        "This chatbot component is now running frontend-only. For full AI features, backend services would be needed.",
        "Check out the book chapters for comprehensive information on Humanoid Robotics and AI."
      ];

      const randomResponse = botResponses[Math.floor(Math.random() * botResponses.length)];
      const botMessage = {
        id: Date.now() + 1,
        text: randomResponse,
        sender: 'bot'
      };
      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    }, 1000);
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