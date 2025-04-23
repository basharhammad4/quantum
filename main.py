# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from quantum.app.routes.predict import router as predict_router

app = FastAPI(
    title="Quantum Shield API",
    description="An API to detect adversarial attacks in images using a hybrid Quantum-Classical deep learning model.",
    version="1.0.0"
)

# ✅ Enable CORS (optional: tweak origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include prediction route
app.include_router(predict_router, prefix="/api")

# ✅ Optional root route
@app.get("/")
def read_root():
    return {"message": "Welcome to Quantum Shield API. Visit /api/predict to use the model."}
