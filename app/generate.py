# app/generate.py

def generate_text(prompt: str, model_name: str) -> str:
    # Example placeholder logic
    if model_name == "finetuned":
        return f"Finetuned GPT-2 response to: {prompt}"
    elif model_name == "pretrained":
        return f"Pretrained GPT-2 response to: {prompt}"
    elif model_name == "earlyexit":
        return f"Early-exit GPT-2 response to: {prompt}"
    else:
        raise ValueError(f"Unknown model: {model_name}")
