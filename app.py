from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Fruit keywords to filter from ImageNet labels ──────────────
FRUIT_KEYWORDS = [
    "banana", "apple", "orange", "lemon", "lime", "strawberry",
    "pineapple", "mango", "grape", "watermelon", "cherry", "peach",
    "pear", "plum", "coconut", "fig", "pomegranate", "kiwi",
    "raspberry", "blueberry", "avocado", "papaya", "guava",
    "tangerine", "clementine", "mandarin", "jackfruit", "durian",
    "cantaloupe", "honeydew", "nectarine", "apricot", "passion fruit",
    "lychee", "persimmon", "quince", "mulberry", "gooseberry",
    "cranberry", "blackberry", "boysenberry", "elderberry",
]

def is_fruit(label: str) -> bool:
    label_lower = label.lower().replace("_", " ")
    return any(fruit in label_lower for fruit in FRUIT_KEYWORDS)

def clean_label(label: str) -> str:
    """Convert ImageNet label like 'Granny_Smith' → 'Granny Smith Apple'"""
    label = label.replace("_", " ").strip()
    # Some ImageNet fruit labels need a hint
    overrides = {
        "granny smith": "Granny Smith Apple",
        "banana":       "Banana",
        "pineapple":    "Pineapple",
        "strawberry":   "Strawberry",
        "orange":       "Orange",
        "lemon":        "Lemon",
        "fig":          "Fig",
        "pomegranate":  "Pomegranate",
        "jackfruit":    "Jackfruit",
        "durian":       "Durian",
        "coconut":      "Coconut",
        "custard apple":"Custard Apple",
    }
    low = label.lower()
    for k, v in overrides.items():
        if k in low:
            return v
    return label.title()


@app.on_event("startup")
def load_model():
    """Download MobileNetV2 pretrained weights on first startup (cached after)."""
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        app.state.model            = MobileNetV2(weights="imagenet")
        app.state.preprocess_input = preprocess_input
        app.state.decode_predictions = decode_predictions
        print("✅ MobileNetV2 loaded (pretrained on ImageNet).")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize and preprocess image for MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)                      # (1, 224, 224, 3)
    arr = app.state.preprocess_input(arr)                  # MobileNetV2 normalization
    return arr


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        image_bytes = await file.read()
        input_arr   = preprocess_image(image_bytes)

        preds = app.state.model.predict(input_arr)                    # (1, 1000)
        top   = app.state.decode_predictions(preds, top=20)[0]        # top-20 labels

        # Filter to fruit-related predictions
        fruit_preds = [
            {"label": clean_label(label), "confidence": float(score)}
            for (_, label, score) in top
            if is_fruit(label)
        ]

        # If no fruit found in top-20, return top result with a note
        if not fruit_preds:
            _, label, score = top[0]
            return {
                "prediction":      clean_label(label),
                "confidence":      float(score),
                "top_predictions": [
                    {"label": clean_label(l), "confidence": float(s)}
                    for (_, l, s) in top[:5]
                ],
                "note": "No fruit detected. Try a clearer fruit photo."
            }

        best = fruit_preds[0]
        return {
            "prediction":      best["label"],
            "confidence":      best["confidence"],
            "top_predictions": fruit_preds[:5],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Fruit Image Recognition API is running 🍎"}
