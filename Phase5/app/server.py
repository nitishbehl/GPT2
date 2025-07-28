from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Ensure root dir is in sys.path to import generate_text.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_text import generate_text  # import your function

app = FastAPI()

origins = [
    "http://localhost:5174",  # frontend URL
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Use /generate endpoint to generate text."}

class GenerateRequest(BaseModel):
    prompt: str
    model: str  # expect 'pretrained', 'finetuned', or 'earlyexit'

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        print(f"Incoming request: prompt={request.prompt}, model={request.model}")
        output = generate_text(request.prompt, request.model)
        return {"prompt": request.prompt, "generated": output}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

