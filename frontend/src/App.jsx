import React, { useState } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('text');
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  // Targets localhost explicitly to cleanly align with browser-side origin bindings
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setOutput('');
    
    const endpoint = activeTab === 'text' ? '/api/generate' : '/api/sentiment';

    // Formats path combinations dynamically to eliminate trailing/missing forward slashes
    const cleanUrl = `${API_BASE_URL.replace(/\/$/, '')}${endpoint}`;

    try {
      const response = await fetch(cleanUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: input }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Something went wrong');
      }

      if (activeTab === 'text') {
        setOutput(data.result);
      } else {
        // Format sentiment response gracefully
        const topResult = data.result; 
        setOutput(`Label: ${topResult.label} | Confidence: ${(topResult.score * 100).toFixed(2)}%`);
      }
    } catch (error) {
      setOutput(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>AI Model Hub</h1>
        <p>Powered by FastAPI & Hugging Face</p>
      </header>

      <div className="tab-buttons">
        <button 
          className={activeTab === 'text' ? 'active' : ''} 
          onClick={() => { setActiveTab('text'); setOutput(''); setInput(''); }}
        >
          Text Generation
        </button>
        <button 
          className={activeTab === 'sentiment' ? 'active' : ''} 
          onClick={() => { setActiveTab('sentiment'); setOutput(''); setInput(''); }}
        >
          Sentiment Analysis
        </button>
      </div>

      <main className="main-content">
        <form onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={activeTab === 'text' ? "Ask the AI anything..." : "Enter text to analyze mood..."}
            rows={4}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Submit Task'}
          </button>
        </form>

        {output && (
          <div className="output-box">
            <h3>Result:</h3>
            <p>{output}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;