from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import json


router = APIRouter()


@router.get("/widget", response_class=HTMLResponse)
def get_widget():
    """
    Return the HTML for the chatbot widget that can be embedded via iframe
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chatbot Widget</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            #chat-container {
                max-width: 400px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                height: 600px;
            }
            #chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }
            .message {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 8px;
                max-width: 80%;
            }
            .user-message {
                align-self: flex-end;
                background-color: #007bff;
                color: white;
            }
            .bot-message {
                align-self: flex-start;
                background-color: #e9ecef;
                color: black;
            }
            #input-area {
                display: flex;
                padding: 15px;
                border-top: 1px solid #eee;
            }
            #user-input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-right: 10px;
            }
            #send-btn {
                padding: 10px 15px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            #send-btn:disabled {
                background-color: #cccccc;
            }
            .loading {
                font-style: italic;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="chat-messages">
                <div class="message bot-message">
                    Hello! I'm your AI textbook assistant. How can I help you with the textbook content today?
                </div>
            </div>
            <div id="input-area">
                <input type="text" id="user-input" placeholder="Ask a question about the textbook..." />
                <button id="send-btn">Send</button>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            
            function addMessage(text, isUser) {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('message');
                messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
                messageDiv.textContent = text;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            async function sendQuestion() {
                const question = userInput.value.trim();
                if (!question) return;
                
                // Add user message to chat
                addMessage(question, true);
                userInput.value = '';
                sendBtn.disabled = true;
                
                // Show loading message
                const loadingDiv = document.createElement('div');
                loadingDiv.classList.add('message', 'bot-message', 'loading');
                loadingDiv.textContent = 'Thinking...';
                chatMessages.appendChild(loadingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                try {
                    // Send question to backend API
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            context_mode: 'full_book'
                        })
                    });
                    
                    // Remove loading message
                    loadingDiv.remove();
                    
                    if (response.ok) {
                        const data = await response.json();
                        addMessage(data.answer, false);
                    } else {
                        const errorData = await response.json();
                        addMessage('Sorry, I couldn\'t answer that: ' + errorData.detail, false);
                    }
                } catch (error) {
                    // Remove loading message
                    loadingDiv.remove();
                    addMessage('Sorry, there was an error processing your request.', false);
                    console.error('Error:', error);
                } finally {
                    sendBtn.disabled = false;
                }
            }
            
            sendBtn.addEventListener('click', sendQuestion);
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/sdk.js")
def get_sdk():
    """
    Return the JavaScript SDK for embedding the chatbot via script tag
    """
    js_content = """
    // Chatbot Widget SDK
    class ChatbotWidget {
        constructor(config) {
            this.config = config;
            this.init();
        }
        
        init() {
            // Create the iframe element
            const iframe = document.createElement('iframe');
            iframe.src = this.config.apiUrl + '/api/widget';
            iframe.width = '400px';
            iframe.height = '600px';
            iframe.style.border = 'none';
            iframe.style.position = 'fixed';
            iframe.style.bottom = '20px';
            iframe.style.right = '20px';
            iframe.style.zIndex = '10000';
            
            // Add the iframe to the specified container or to the body
            const container = document.getElementById(this.config.containerId);
            if (container) {
                container.appendChild(iframe);
            } else {
                document.body.appendChild(iframe);
            }
        }
        
        static init(config) {
            new ChatbotWidget(config);
        }
    }
    
    // Make it available globally
    window.ChatbotWidget = ChatbotWidget;
    """
    return js_content