import { useState } from 'react';
import './index.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [model, setModel] = useState('pretrained');

  const generateText = async () => {
    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model })
      });
      const data = await res.json();
      console.log(data); // <-- check browser console
      setOutput(data.output || '⚠️ No output received');
    } catch (err) {
      console.error('Fetch error:', err);
      setOutput(' Error contacting backend');
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 mt-10">
      <h1 className="text-3xl font-bold">GPT-2 Comparison Tool</h1>

      <textarea
        className="border w-1/2 p-2"
        rows={4}
        placeholder="Enter your prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <select
        className="border p-2"
        value={model}
        onChange={(e) => setModel(e.target.value)}
      >
        <option value="pretrained">Pretrained</option>
        <option value="finetuned">Fine-Tuned</option>
        <option value="earlyexit">Early Exit</option>
      </select>

      <button onClick={generateText} className="bg-blue-600 text-white px-4 py-2 rounded">
        Generate
      </button>

      {/* OUTPUT */}
      {output && (
        <div className="w-1/2 mt-6">
          <h2 className="text-xl font-semibold mb-2">Generated Output:</h2>
          <pre className="bg-gray-100 p-4 whitespace-pre-wrap border rounded">
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;
