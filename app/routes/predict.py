from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
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

# ✅ Load the model (absolute path works well on Render)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maryam12.h5'))
model = load_model(MODEL_PATH, compile=False)

# ✅ Initialize FastAPI router
router = APIRouter()

# ✅ NPY file prediction route
@router.post("/predict-npy/")
async def predict_npy(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = np.load(io.BytesIO(contents), allow_pickle=True)

        # Unwrap if stored as a dict
        if isinstance(data, dict) and "image" in data:
            data = data["image"]

        if not isinstance(data, np.ndarray):
            return JSONResponse(content={"error": "Uploaded file is not a valid numpy array."}, status_code=400)

        if data.shape != (256, 512, 3):
            return JSONResponse(content={"error": f"Invalid image shape: {data.shape}"}, status_code=400)

        # Expand dims to match batch shape
        input_data = np.expand_dims(data, axis=0)

        prediction = model.predict(input_data)
        label = "attack" if prediction[0][0] > 0.5 else "clean"
        confidence = float(prediction[0][0]) if label == "attack" else 1 - float(prediction[0][0])

        return JSONResponse(content={
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
