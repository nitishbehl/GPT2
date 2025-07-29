import React, { useState } from 'react';

const App = () => {
  const [input, setInput] = useState('');
  const [model, setModel] = useState('pretrained');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generateText = async () => {
    if (!input.trim()) {
      setOutput("⚠️ Please enter a prompt.");
      return;
    }

    setLoading(true);
    setError('');
    setOutput('');

    try {
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: input, model }),
      });

      const data = await response.json();
      if (response.ok) {
        setOutput(data.output || "⚠️ No output returned.");
      } else {
        setError(data.detail || "❌ Something went wrong.");
      }
    } catch (err) {
      setError("❌ Error: " + err.message);
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-2xl mx-auto bg-white shadow p-6 rounded-xl">
        <h1 className="text-2xl font-bold mb-4 text-center">GPT-2 Comparison Tool</h1>

        <textarea
          rows="4"
          className="w-full border p-2 mb-4 rounded"
          placeholder="Enter your prompt..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />

        <select
          className="w-full border p-2 mb-4 rounded"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="pretrained">Pretrained</option>
          <option value="finetuned">Fine-tuned</option>
          <option value="earlyexit">Early Exit</option>
        </select>

        <button
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded"
          onClick={generateText}
          disabled={loading}
        >
          {loading ? "Generating..." : "Generate"}
        </button>

        {error && (
          <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">{error}</div>
        )}

        {output && (
          <div className="mt-6 bg-gray-50 border border-gray-300 rounded p-4">
            <strong className="block text-gray-700 mb-2">Output:</strong>
            <pre className="whitespace-pre-wrap text-gray-900">{output}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;

