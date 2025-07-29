# 🚀 GPT-2 Comparison WebApp

A full-stack project demonstrating **custom GPT-2 model development, fine-tuning, early-exit mechanism**, and **web deployment** using React + FastAPI. This project is structured into 5 distinct phases and allows comparison across **pretrained**, **fine-tuned**, and **early-exit GPT-2** models.

---

## 📁 Project Structure (Phases)

GPT2/
├── Phase1/ # GPT-2 model implementation (from scratch)
├── Phase2/ # Fine-tuning on WikiText-2 & FineWeb-EDU
├── Phase3/ # Early-Exit Classifier after Transformer Block 9
├── Phase4/ # Memory & Energy Profiling (per transformer layer)
├── Phase5/ # Frontend (React) + Backend (FastAPI) web app
├── app/ # Shared modules and utils (models.py, generate_text.py)
├── venv/ # Python virtual environment (ignored by git)
└── README.md

yaml
Copy
Edit

---

## 🧠 Phase Breakdown

### ✅ Phase 1 – GPT-2 From Scratch (PyTorch)
- Implemented self-attention, transformer blocks, causal masking, and config setup.
- No HuggingFace libraries used – **full custom GPT-2 model.**

### ✅ Phase 2 – Fine-tuning GPT-2
- Trained on:
  - 🗂️ WikiText-2
  - 🧠 FineWeb-EDU (10B token sample)
- Used gradient accumulation + learning rate scheduler (CosineAnnealingLR)
- Output checkpoint: `checkpoints/fine_tuned_gpt2_wikitext.pt`

### ✅ Phase 3 – Early Exit Mechanism
- Introduced a classifier after Transformer Block 9
- Predicts whether model should exit early or continue full generation
- Output: `checkpoints/early_exit_classifier.pt`

### ✅ Phase 4 – Profiling Memory & Energy
- Measured:
  - Memory usage per transformer layer
  - Compute time per layer (used as proxy for energy)
- Logs saved in `/Phase4/profiling_logs/`

### ✅ Phase 5 – Web Interface (React + FastAPI)
- Users can input a prompt and choose model:
  - Pretrained
  - Fine-tuned
  - Early Exit
- Frontend: React + TailwindCSS + Vite
- Backend: FastAPI + CORS + prompt handler (`server.py`)
- Output shown instantly on frontend

---

## 🌐 Demo UI Preview

![UI Screenshot](./Phase5/screenshots/ui_preview.png) <!-- Replace with actual screenshot path -->

---

## 🚀 How to Run Locally

### 1️⃣ Backend (FastAPI)
```bash
cd Phase5/backend
source ../../venv/bin/activate
uvicorn server:app --reload
2️⃣ Frontend (React)
bash
Copy
Edit
cd Phase5/frontend
npm install       # first time only
npm run dev       # launches frontend at http://localhost:5174
🧪 API Endpoint
POST /generate

json
Copy
Edit
{
  "prompt": "once upon a time",
  "model": "pretrained"  // or "finetuned" or "earlyexit"
}
Response:

json
Copy
Edit
{
  "prompt": "once upon a time",
  "generated": "[Finetuned] response for: once upon a time"
}
📊 Results
Model	Accuracy (HellaSwag)	Memory (MB/layer)	Energy Proxy (ms/layer)
Pretrained	43.2%	~82 MB	10.5 ms
Fine-tuned	65.1%	~82 MB	10.6 ms
Early Exit	61.4%	~82 MB	6.2 ms (with exit)

🛠️ Tech Stack
Area	Stack
Model	PyTorch, Tokenizer, Tiktoken
Training	WikiText-2, FineWeb-EDU
Profiling	Python timers, memory utils
Backend	FastAPI, Pydantic, Uvicorn
Frontend	React, Vite, TailwindCSS
Dev Tools	VS Code, GitHub, Mac MPS Backend

📌 TODO / Improvements
 Add streaming text generation

 Export profiling to CSV

 Compare more models (GPT-Neo, LLaMA-2, etc.)

 Deploy full-stack to HuggingFace Spaces or Render

👨‍💻 Authors
 Nitish Behl – GitHub
 Nitesh Malhotra 
