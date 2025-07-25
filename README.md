# GPT-2: Understanding, Fine-Tuning, and Profiling with Early Exit & Web Deployment

This project explores a complete pipeline of working with GPT-2 from scratch using PyTorch. It is divided into five well-defined phases, each focusing on a critical aspect of building, training, optimizing, and deploying a language model.

## 📁 Project Phases

### ✅ Phase 1: Custom GPT-2 Implementation
- Built GPT-2 from scratch using PyTorch.
- Implemented transformer blocks, causal self-attention, and generation loop.
- Tokenized datasets using `tiktoken` (OpenAI's tokenizer).

### ✅ Phase 2: Fine-Tuning GPT-2
- Fine-tuned GPT-2 on the **FineWeb-EDU** and **Wikitext-2** datasets.
- Used gradient accumulation, learning rate scheduler, and loss tracking.
- Saved final model checkpoints (now excluded from GitHub).

### ✅ Phase 3: Early-Exit Classifier
- Added an early-exit classifier after block 9 of the transformer.
- The classifier can predict the next token earlier, reducing compute.
- Trained on hidden states from the fine-tuned GPT-2.

### ✅ Phase 4: Profiling (Memory & Compute)
- Profiled each transformer layer for:
  - Time per forward pass
  - Memory usage (on MPS backend)
- Used for deciding optimal early-exit layers.

### ✅ Phase 5: Web App Deployment
- Built a web app using **React + Vite + Tailwind**.
- Compares:
  - Pretrained GPT-2
  - Fine-Tuned GPT-2
  - Early-Exit GPT-2
- Backend: Python FastAPI serving model outputs.

## 🧪 Evaluation
- Integrated **HellaSwag** benchmark for comparing performance and accuracy.
- Used multiple prompts to observe prediction behavior and compute efficiency.

## 🚫 Excluded Files
The following large files were removed using `git filter-repo` to comply with GitHub size limits:
- Model checkpoints (`*.pt`) larger than 100 MB.
- `.gitignore` handles excluding these files.

## 📦 Installation

```bash
# Python
pip install -r requirements.txt

# Frontend
cd frontend
npm install
npm run dev
uvicorn server:app --reload --port 8000


🤖 Authors
Nitish Behl

Nitesh Malhotra
(School of Computer Science and Technology, Algoma University)

📄 Citation (HellaSwag Dataset)
graphql
Copy
Edit
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
yaml
Copy
Edit
