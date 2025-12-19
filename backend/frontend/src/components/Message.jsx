import React from 'react';
import './Message.css';

const Message = ({ message }) => {
    if (message.loading) {
        return (
            <div className={`message message-bot`}>
                <div className="message-content">
                    <div className="loading-dots">
                        <span>.</span>
                        <span>.</span>
                        <span>.</span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className={`message message-${message.sender}`}>
            <div className="message-content">
                {message.text}
                {message.sources && message.sources.length > 0 && (
                    <div className="sources">
                        <strong>Sources:</strong>
                        <ul>
                            {message.sources.map((source, index) => (
                                <li key={index}>{source}</li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Message;