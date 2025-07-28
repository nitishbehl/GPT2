from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "pretrained"  # default to pretrained

def generate_text(prompt: str, model: str) -> str:
    # Replace this with your actual generation logic per model
    if model == "pretrained":
        return f"[Pretrained] response for: {prompt}"
    elif model == "finetuned":
        return f"[finetuned] response for: {prompt}"
    elif model == "earlyexit":
        return f"[Early Exit] response for: {prompt}"
    else:
        raise ValueError(f"Unknown model: {model}")

@app.post("/generate")
async def generate(request: GenerateRequest):
    model = request.model.lower()
    if model not in ["pretrained", "finetuned", "earlyexit"]:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'")

    generated_text = generate_text(request.prompt, model)
    return {"prompt": request.prompt, "generated": generated_text}

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
