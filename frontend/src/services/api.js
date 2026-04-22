const API_BASE_URL = 'https://shrut04-fino-backend-api.hf.space';
// const API_BASE_URL = 'http://localhost:8000';

export const FinoAPI = {
  checkHealth: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) return false;
      const data = await response.json();
      return data.status === 'healthy' || data.graph_loaded;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  },

  sendMessage: async (query) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      return data.answer;
    } catch (error) {
      console.error('Send message failed:', error);
      throw error;
    }
  }
};