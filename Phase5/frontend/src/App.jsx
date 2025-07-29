import React, { useState } from 'react';

function App() {
  const [input, setInput] = useState('');
  const [model, setModel] = useState('pretrained');
  const [output, setOutput] = useState('');
  const [profiling, setProfiling] = useState([]);
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
    setProfiling([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input, model })
      });

      const data = await response.json();

      if (response.ok) {
        setOutput(data.output || "⚠️ No output returned.");
        setProfiling(data.profiling || []);
      } else {
        setError(data.detail || "❌ Something went wrong.");
      }
    } catch (err) {
      setError("❌ Error: " + err.message);
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-3xl mx-auto bg-white shadow-lg rounded-xl p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">GPT-2 Comparison Tool</h1>

        <textarea
          rows="4"
          className="w-full border border-gray-300 p-3 rounded mb-4"
          placeholder="Enter your prompt here..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />

        <select
          className="w-full border border-gray-300 p-2 rounded mb-4"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="pretrained">Pretrained</option>
          <option value="finetuned">Fine-tuned</option>
          <option value="earlyexit">Early Exit</option>
        </select>

        <button
          onClick={generateText}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 w-full rounded"
          disabled={loading}
        >
          {loading ? "Generating..." : "Generate Text"}
        </button>

        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        {output && (
          <div className="mt-6 p-4 bg-gray-50 border rounded">
            <h2 className="font-semibold text-lg mb-2">Output ({model} model):</h2>
            <p className="whitespace-pre-wrap">{output}</p>
          </div>
        )}

        {profiling.length > 0 && model !== 'earlyexit' && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Profiling Info (per layer)</h2>
            <table className="table-auto w-full text-sm border border-gray-300">
              <thead>
                <tr className="bg-gray-200">
                  <th className="border px-2 py-1">Layer</th>
                  <th className="border px-2 py-1">Memory (MB)</th>
                  <th className="border px-2 py-1">Time (ms)</th>
                </tr>
              </thead>
              <tbody>
                {profiling.map((p, idx) => (
                  <tr key={idx} className="text-center">
                    <td className="border px-2 py-1">{p.layer}</td>
                    <td className="border px-2 py-1">{(p.memory_mb ?? 0).toFixed(2)}</td>
                    <td className="border px-2 py-1">{(p.time_ms ?? 0).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
