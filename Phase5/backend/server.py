from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .generate_text import generate_text

app = FastAPI()

# Allow all CORS (for dev only; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "pretrained"

@app.post("/generate")
async def generate_handler(request: GenerateRequest):
    model = request.model.lower()
    if model not in ["pretrained", "finetuned", "earlyexit"]:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'")

    try:
        generated = generate_text(prompt=request.prompt, model=model)
        return {"prompt": request.prompt, "output": generated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Only needed when running standalone (not with `uvicorn server:app --reload`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
