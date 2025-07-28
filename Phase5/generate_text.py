import sys
import os
import torch
import torch.nn.functional as F
import tiktoken

# Add Phase1 directory to sys.path to import GPT model code
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase1")
from train_gpt2 import GPT, GPTConfig

device = "mps" if torch.backends.mps.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# Paths to your saved checkpoints for each model
MODEL_PATHS = {
    "finetuned": "/Users/nitishbehl/Desktop/gpt2_project/Phase2/checkpoints/fine_tuned_gpt2_wikitext.pt",
    "pretrained": None,  # You can load pretrained weights if available
    "earlyexit": "/Users/nitishbehl/Desktop/gpt2_project/Phase3/checkpoints/early_exit_model.pt",  # update path accordingly
}

# Cache loaded models here
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
    model_instance = GPT(config)

    checkpoint_path = MODEL_PATHS.get(model_name)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_instance.load_state_dict(state_dict, strict=False)
    else:
        # If no checkpoint (e.g., pretrained), consider loading default weights or skip
        print(f"[Warning] No checkpoint for model '{model_name}', using random initialized weights.")

    model_instance.to(device)
    model_instance.eval()

    _loaded_models[model_name] = model_instance
    return model_instance

def generate_text(prompt="Hello, I am a language model,", model="finetuned", max_length=50):
    model_name = model.lower()
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model}")

    model_instance = load_model(model_name)

    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        while x.size(1) < max_length:
            logits, _ = model_instance(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    return enc.decode(x[0].tolist())
