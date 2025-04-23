from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

# Load model once when the router is imported
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'maryam12.h5')
model = load_model(MODEL_PATH, compile=False)

router = APIRouter()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((512, 256))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)

        prediction = model.predict(img_array)
        label = "attack" if prediction[0][0] > 0.5 else "clean"
        confidence = float(prediction[0][0]) if label == "attack" else 1 - float(prediction[0][0])

        return JSONResponse(content={
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
