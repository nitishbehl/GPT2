import sys
import torch
import torch.nn.functional as F
import tiktoken
import os

# Select device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Add Phase paths
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase1")
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase3")

# Import model components
from train_gpt2 import GPT, GPTConfig
from early_exit import EarlyExitClassifier

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load early-exit classifier
early_exit_classifier = EarlyExitClassifier(hidden_size=384, vocab_size=50257)
early_exit_ckpt = "/Users/nitishbehl/Desktop/gpt2_project/Phase3/checkpoints/early_exit_classifier.pt"
early_exit_classifier.load_state_dict(torch.load(early_exit_ckpt, map_location=device))
early_exit_classifier.to(device)
early_exit_classifier.eval()

# Define model paths
MODEL_PATHS = {
    "finetuned": "/Users/nitishbehl/Desktop/gpt2_project/Phase2/checkpoints/fine_tuned_gpt2_wikitext.pt",
    "pretrained": None,  # You may add pretrained weights here if available
    "earlyexit": None,   # Reuses finetuned + early exit head
}

_loaded_models = {}

def load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    config = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=6,
        n_embd=384
    )
    model = GPT(config)

    ckpt_path = MODEL_PATHS.get(model_name)
    if ckpt_path:
        print(f"[INFO] Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    elif model_name != "earlyexit":
        print(f"[WARNING] No checkpoint for '{model_name}', using randomly initialized weights.")

    model.to(device)
    model.eval()
    _loaded_models[model_name] = model
    return model

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., top_k:] = 1  # Only keep top_k tokens
    sorted_logits[sorted_indices_to_remove] = -float('Inf')

    probs = F.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, next_token)

def generate_text(prompt="Hello", model="finetuned", max_length=80, temperature=0.9, top_k=50, top_p=0.95):
    model_name = model.lower()
    print(f"[DEBUG] Prompt: {prompt} | Model: {model_name}")

    if model_name not in ["pretrained", "finetuned", "earlyexit"]:
        raise ValueError(f"Unknown model: {model_name}")

    gpt_model = load_model(model_name if model_name != "earlyexit" else "finetuned")

    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        while x.size(1) < max_length:
            if model_name == "earlyexit":
                logits, all_hidden = gpt_model(x, return_all_hidden=True)
                hidden9 = all_hidden[9]
                logits = early_exit_classifier(hidden9)
            else:
                logits, _ = gpt_model(x)

            logits = logits[:, -1, :] / temperature
            next_token = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            x = torch.cat((x, next_token), dim=1)

    decoded = enc.decode(x[0].tolist())
    print(f"[DEBUG] Output: {decoded}")
    return decoded
