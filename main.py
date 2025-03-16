from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

# ✅ CORS middleware should be **before defining routes**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ✅ Allow requests from frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load API Key (Avoid hardcoding)
GENAI_API_KEY = os.getenv("GENAI_API_KEY")  # Replace with real key or use .env
if not GENAI_API_KEY:
    raise ValueError("❌ Google Gemini API key is missing! Set GENAI_API_KEY in .env or environment variables.")

# ✅ Configure Google AI API
genai.configure(api_key=GENAI_API_KEY)

# ✅ Check available models (useful for debugging)
try:
    models = genai.list_models()
    available_models = [model.name for model in models]
    print("✅ Available models:", available_models)
except Exception as e:
    print("❌ Error fetching model list:", e)
    available_models = []

@app.get("/")
def read_root():
    return {"message": "Welcome to the GenAI Learning Games API!"}

@app.get("/generate_question")
def generate_question(topic: str = "math"):
    """Generates a learning question using Google Gemini AI."""
    if not GENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Google Gemini API key is missing!")

    try:
        model_name = "models/gemini-1.5-flash"
        if model_name not in available_models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found. Check your API access.")

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(f"Generate a {topic} learning question.")
        return {"question": response.text.strip()}

    except genai.types.RateLimitError:
        raise HTTPException(status_code=429, detail="Quota exceeded. Please try again later.")

    except genai.types.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ✅ Define request model for answer submission
class AnswerRequest(BaseModel):
    question: str
    answer: str

@app.post("/evaluate_answer")
def evaluate_answer(request: AnswerRequest):
    """Evaluates the user's answer using Google Gemini AI."""
    if not GENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Google Gemini API key is missing!")

    try:
        model_name = "models/gemini-1.5-flash"
        if model_name not in available_models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found. Check your API access.")

        model = genai.GenerativeModel(model_name)
        prompt = (
    f"Here is a learning question: {request.question}\n"
    f"The user's answer: {request.answer}\n"
    "Evaluate the correctness of this answer. If incorrect, provide a helpful hint or explanation."
)
        response = model.generate_content(prompt)
        return {"feedback": response.text.strip()}

    except genai.types.RateLimitError:
        raise HTTPException(status_code=429, detail="Quota exceeded. Try again later.")

    except genai.types.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
