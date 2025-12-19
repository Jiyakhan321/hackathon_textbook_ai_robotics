class ApiService {
    constructor(apiUrl, apiKey) {
        this.apiUrl = apiUrl;
        this.apiKey = apiKey;
    }

    async queryFullBook(question) {
        try {
            const response = await fetch(`${this.apiUrl}/api/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.apiKey
                },
                body: JSON.stringify({
                    question: question,
                    context_mode: 'full_book'
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error querying full book:', error);
            throw error;
        }
    }

    async querySelectedText(question, selectedText) {
        try {
            const response = await fetch(`${this.apiUrl}/api/query/selected-text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.apiKey
                },
                body: JSON.stringify({
                    question: question,
                    selected_text: selectedText,
                    context_mode: 'selected_text'
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error querying selected text:', error);
            throw error;
        }
    }

    async getWidgetHtml() {
        try {
            const response = await fetch(`${this.apiUrl}/api/widget`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.text();
        } catch (error) {
            console.error('Error getting widget HTML:', error);
            throw error;
        }
    }
}

export default ApiService;