import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import InputField from './InputField';
import { getApiConfig } from '../config/apiConfig';
import './ChatWidget.css';

const ChatWidget = ({ apiKey, apiUrl }) => {
    const [messages, setMessages] = useState([
        { id: 1, text: "Hello! I'm your AI textbook assistant. How can I help you today?", sender: 'bot' }
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (text, contextMode = 'full_book', selectedText = null) => {
        if (!text.trim() || isLoading) return;

        // Add user message to the chat
        const userMessage = {
            id: Date.now(),
            text: text,
            sender: 'user'
        };
        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        try {
            // Prepare the request based on context mode
            let requestData;
            if (contextMode === 'selected_text' && selectedText) {
                requestData = {
                    question: text,
                    selected_text: selectedText,
                    context_mode: 'selected_text'
                };
            } else {
                requestData = {
                    question: text,
                    context_mode: 'full_book'
                };
            }

            // Call the backend API
            const response = await fetch(`${apiUrl}/api/query${contextMode === 'selected_text' ? '/selected-text' : ''}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get response');
            }

            const data = await response.json();

            // Add bot response to the chat
            const botMessage = {
                id: Date.now() + 1,
                text: data.answer,
                sender: 'bot',
                sources: data.sources || []
            };
            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Error sending message:', error);

            // Add error message to the chat
            const errorMessage = {
                id: Date.now() + 1,
                text: `Sorry, I couldn't process your request: ${error.message}`,
                sender: 'bot'
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-widget">
            <div className="chat-header">
                <h3>AI Textbook Assistant</h3>
            </div>
            <div className="chat-messages">
                {messages.map((message) => (
                    <Message key={message.id} message={message} />
                ))}
                {isLoading && <Message message={{ id: 'loading', text: 'Thinking...', sender: 'bot', loading: true }} />}
                <div ref={messagesEndRef} />
            </div>
            <InputField onSendMessage={handleSendMessage} />
        </div>
    );
};

export default ChatWidget;