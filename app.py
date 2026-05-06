from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
import subprocess

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_FILE = "model.pkl"
MAPPING_FILE = "mapping.pkl"

class FruitInput(BaseModel):
    mass: float
    width: float
    height: float
    color_score: float

# Load model on startup
@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Model not found! Training automatically...")
        try:
            # Use subprocess to run the training script
            subprocess.run(["python3", "model_train.py"], check=True)
        except Exception as e:
            print(f"Error training model: {e}")
    
    if os.path.exists(MODEL_FILE):
        try:
            app.state.model = joblib.load(MODEL_FILE)
            app.state.mapping = joblib.load(MAPPING_FILE)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Warning: Model could not be loaded or created.")

@app.post("/predict")
async def predict(data: FruitInput):
    # Check if model is loaded
    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input for prediction
        features = [[data.mass, data.width, data.height, data.color_score]]
        
        # Get prediction
        label = app.state.model.predict(features)[0]
        prediction = app.state.mapping.get(label, "Unknown")
        
        # Get confidence (if supported by model)
        # Note: KNN supports predict_proba
        probabilities = app.state.model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Fruit Recognition API is running"}
