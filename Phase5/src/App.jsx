import React, { useState } from 'react';

// Set your backend API base URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('finetuned'); // Default model

  // Handle form submit
  const handleSubmit = (e) => {
    e.preventDefault();
    generateText();
  };

  const generateText = async () => {
    if (!prompt.trim()) {
      setOutput(' Please enter a prompt.');
      return;
    }
    if (!model) {
      setOutput(' Please select a model.');
      return;
    }

    setLoading(true);
    setOutput(''); // Clear previous output

    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt, // matches FastAPI: prompt
          model: model,   // matches FastAPI: model
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        const message = data?.error || `Error ${response.status}`;
        setOutput(` ${message}`);
        return;
      }

      if (data && data.generated) {
        setOutput(data.generated);
      } else {
        setOutput('⚠️ Unexpected response from the server.');
      }
    } catch (error) {
      console.error('Request failed:', error);
      setOutput(' Error connecting to backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20, fontFamily: "Arial, sans-serif" }}>
      <h2>Text Generation</h2>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Type your prompt here"
          style={{ width: "100%", padding: 8, marginBottom: 10, fontSize: 16 }}
          disabled={loading}
        />

        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          style={{ width: "100%", padding: 8, marginBottom: 10, fontSize: 16 }}
          disabled={loading}
        >
          <option value="finetuned">Finetuned</option>
          <option value="earlyexit">Early-Exit</option>
          <option value="pretrained">Pretrained</option>
        </select>

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            padding: 10,
            fontSize: 16,
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Generating..." : "Generate"}
        </button>
      </form>

      <div style={{ marginTop: 20 }}>
        <h3>Output:</h3>
        <pre style={{ whiteSpace: "pre-wrap", backgroundColor: "#f4f4f4", padding: 10, minHeight: 100 }}>
          {output || <span style={{ color: "#888" }}>No output yet.</span>}
        </pre>
      </div>
    </div>
  );
}

export default App;
