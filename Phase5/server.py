from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os

# Add the path to your generate_text.py if needed
sys.path.append(os.path.dirname(__file__))

# Import the actual function
from generate_text import generate_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "finetuned"

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    prompt = request.prompt
    model = request.model

    print(f"[DEBUG] FastAPI received prompt: {prompt} | model: {model}")
    
    try:
        output = generate_text(prompt, model=model)
        return {"prompt": prompt, "output": output}
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
