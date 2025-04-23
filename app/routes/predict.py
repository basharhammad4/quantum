from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from PIL import Image
import io
import os
import pennylane as qml
from pennylane.qnn import KerasLayer

# ✅ Define quantum layer again for loading
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits, 3)}

# ✅ Register KerasLayer
get_custom_objects().update({
    "KerasLayer": lambda **kwargs: KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
})

# ✅ Correct path to model file
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maryam12.h5'))
model = load_model(MODEL_PATH, compile=False)

# ✅ Initialize the router
router = APIRouter()

# ✅ Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((512, 256))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Predict route
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
