from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import os
import io
import pennylane as qml
from pennylane.qnn import KerasLayer

# ✅ Quantum Layer Configuration
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits, 3)}
get_custom_objects().update({
    "KerasLayer": lambda **kwargs: KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
})

# ✅ Load model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maryam12.h5'))
model = load_model(MODEL_PATH, compile=False)

# ✅ FastAPI router
router = APIRouter()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = np.load(io.BytesIO(contents), allow_pickle=True)

        # If the image is wrapped in a dict (like from `{'image': array}`), extract it
        if isinstance(img_array, dict) and 'image' in img_array:
            img_array = img_array['image']

        if img_array.ndim != 3:
            return JSONResponse(content={"error": "Invalid image shape. Expected 3D array."}, status_code=400)

        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = "attack" if prediction[0][0] > 0.5 else "clean"
        confidence = float(prediction[0][0]) if label == "attack" else 1 - float(prediction[0][0])

        return JSONResponse(content={
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
