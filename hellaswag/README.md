GPT-2 Final Project: Understanding, Fine-Tuning, and Profiling GPT-2 with Early Exit and Web Inference
This project explores the GPT-2 model architecture from scratch in PyTorch, focusing on training, evaluation, optimization, and interactive comparison. It was built as part of a Neural Networks and Deep Learning course final project (Summer 2025), with the following phases:

📌 Project Structure
graphql
Copy
Edit
gpt2_project/
├── Phase1/                # Custom GPT-2 implementation from scratch
├── Phase2/                # Fine-tuning on FineWeb-EDU and Wikitext-2
├── Phase3/                # Early-exit classifier at Transformer Block 9
├── Phase4/                # Profiling: memory & compute usage
├── Phase5/                # Web app for model comparison (pretrained, fine-tuned, early-exit)
├── hellaswag/             # HellaSwag benchmark integration
🧠 Phase 1: GPT-2 Architecture from Scratch
We implemented the full GPT-2 model in PyTorch, including:

Multi-head self-attention

Feedforward blocks

Positional encoding

GPTConfig, TransformerBlock, and GPT classes

Tokenization using tiktoken GPT-2 tokenizer

🔧 Phase 2: Fine-Tuning GPT-2
We fine-tuned the GPT-2 model on two datasets:

✅ FineWeb-EDU (10B tokens, local .parquet format)

✅ Wikitext-2 for testing performance and validation

Training features:

Gradient accumulation

Learning rate scheduler

Tokenized .bin shards

Loss logging and generation output

⚡ Phase 3: Early-Exit GPT-2 (Efficient Inference)
Introduced an early-exit mechanism after Transformer block 9:

A small classifier predicts if early exit is safe

Reduces computation for simple prompts

Trained separately using intermediate hidden states

📊 Phase 4: Profiling Memory and Inference Cost
Measured layer-wise compute and memory usage using:

time.perf_counter() for compute time

MPS backend on macOS

Profile logs captured to CSV

Used to compare standard vs early-exit GPT-2.

🌐 Phase 5: Web App for Model Comparison
A Vite + React frontend with a Python backend (FastAPI) that:

Accepts user prompts

Allows model selection (pretrained, finetuned, earlyexit)

Displays generated output on the fly

Backend loads models and runs generation using the appropriate pipeline.

bash
Copy
Edit
# Run frontend
cd Phase5/frontend
npm install && npm run dev

# Run backend
cd ../
python server.py
📚 HellaSwag Benchmark
Evaluated our fine-tuned and early-exit GPT-2 models on the HellaSwag benchmark:

MCQ-style sentence completion

Measures commonsense reasoning

We achieved improvements with fine-tuning

graphql
Copy
Edit
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of ACL 2019},
    year={2019}
}
✅ Key Features
✅ Full GPT-2 from scratch (no HuggingFace)

✅ Tokenization using tiktoken

✅ Fine-tuning on real datasets

✅ Early exit to reduce latency

✅ Inference profiling

✅ Web interface for model comparison

✅ HellaSwag integration

📎 Requirements
Python 3.9+

PyTorch

tiktoken

numpy, matplotlib

uvicorn, fastapi

vite, react, tailwind

