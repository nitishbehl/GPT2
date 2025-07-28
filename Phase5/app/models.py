import torch
import os
from Phase1.train_gpt2 import GPT, GPTConfig

def load_model(model_type="finetuned"):
    """
    Load either fine-tuned GPT-2 or early-exit classifier.
    """
    if model_type == "finetuned":
        ckpt_path = "../../Phase2/checkpoints/fine_tuned_gpt2.pt"
    elif model_type == "earlyexit":
        ckpt_path = "../../Phase3/checkpoints/early_exit_classifier.pt"
    else:
        raise ValueError("Invalid model type. Use 'finetuned' or 'earlyexit'.")

    # Define model config (must match training config)
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768
    )

    model = GPT(config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model
