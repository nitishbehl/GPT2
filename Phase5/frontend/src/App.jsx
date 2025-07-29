import React, { useState } from 'react';

function App() {
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
        body: JSON.stringify({
          prompt: input,
          model: model,
        }),
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
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded w-full"
          onClick={generateText}
        >
          {loading ? "Generating..." : "Generate"}
        </button>

        {error && (
          <div className="w-full mt-4 p-4 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}

        {output && (
          <div className="w-full max-w-xl mt-6 p-4 bg-white rounded shadow text-black text-lg">
            <strong>Output:</strong>
            <p>{output}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
