from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from .generate import generate_text  # Ensure this file/module exists

app = FastAPI()

# Update these based on your frontend port (keep "*" for testing)
origins = [
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for dev only; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Use /generate endpoint to generate text."}

#  Match field names to what frontend sends
class GenerateRequest(BaseModel):
    prompt: str
    model: str  # e.g. "fine-tuned"

@app.post("/generate")
def generate(request: GenerateRequest):
    model_name = request.model.lower()
    try:
        output = generate_text(request.prompt, model_name)
        return {"prompt": request.prompt, "generated": output}
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
