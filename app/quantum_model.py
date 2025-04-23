import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pennylane as qml
from pennylane.qnn import KerasLayer

# ✅ Suppress TensorFlow & Warning Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ✅ Quantum Circuit Setup
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits, 3)}
q_layer = KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# ✅ Build and Load Model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(256, 512, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_qubits, activation="tanh"),
        q_layer,
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ✅ Load Trained Model
model = build_model()
model.load_weights("/content/drive/MyDrive/GP/QuantumShield_API/app/maryam12.h5")  # Adjust path as needed

# ✅ Inference Function
def predict_attack(image: np.ndarray) -> dict:
    """
    Perform prediction on a single image.

    Args:
        image (np.ndarray): Image of shape (256, 512, 3)

    Returns:
        dict: {'label': 0 or 1, 'confidence': float}
    """
    image = tf.image.resize(image, (256, 512)) / 255.0
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    pred = model.predict(image)[0][0]
    label = int(pred > 0.5)
    return {"label": label, "confidence": float(pred)}
