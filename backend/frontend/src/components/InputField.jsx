import React, { useState } from 'react';
import './InputField.css';

const InputField = ({ onSendMessage }) => {
    const [inputText, setInputText] = useState('');
    const [contextMode, setContextMode] = useState('full_book'); // 'full_book' or 'selected_text'
    const [selectedText, setSelectedText] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!inputText.trim()) return;

        onSendMessage(inputText, contextMode, contextMode === 'selected_text' ? selectedText : null);
        setInputText('');
    };

    return (
        <form className="input-field" onSubmit={handleSubmit}>
            <div className="context-controls">
                <label>
                    <input
                        type="radio"
                        value="full_book"
                        checked={contextMode === 'full_book'}
                        onChange={() => setContextMode('full_book')}
                    />
                    Full Book
                </label>
                <label>
                    <input
                        type="radio"
                        value="selected_text"
                        checked={contextMode === 'selected_text'}
                        onChange={() => setContextMode('selected_text')}
                    />
                    Selected Text
                </label>
            </div>

            {contextMode === 'selected_text' && (
                <textarea
                    className="selected-text-input"
                    value={selectedText}
                    onChange={(e) => setSelectedText(e.target.value)}
                    placeholder="Paste the text you want to ask about..."
                    rows="3"
                />
            )}

            <div className="input-area">
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Ask a question about the textbook..."
                    disabled={contextMode === 'selected_text' && !selectedText.trim()}
                />
                <button type="submit" disabled={!inputText.trim()}>Send</button>
            </div>
        </form>
    );
};

export default InputField;