from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from predictor import BrainTumorPredictor
from config import (
    CLASS_NAMES, MODEL_VERSION, API_VERSION,
    MAX_FILE_SIZE_MB, ALLOWED_TYPES
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading Brain Tumor model...")
    BrainTumorPredictor.get_instance()
    print("[Startup] Model ready. API is live.")
    yield
    print("[Shutdown] Cleaning up...")

# ── App init ────────────────────────────────────────────────
app = FastAPI(
    title       = "Brain Tumor AI API",
    description = "VGG16-based 4-class brain tumor classification from MRI images",
    version     = API_VERSION,
    lifespan    = lifespan
)

# ── CORS — allow Next.js frontend ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service" : "Brain Tumor AI API",
        "version" : API_VERSION,
        "status"  : "running",
        "docs"    : "/docs"
    }


@app.get("/health")
def health():
    predictor = BrainTumorPredictor.get_instance()
    return {
        "status"        : "healthy",
        "model_loaded"  : predictor.model is not None,
        "model_version" : MODEL_VERSION,
        "classes"       : CLASS_NAMES
    }


@app.get("/model-info")
def model_info():
    return {
        "model_version"   : MODEL_VERSION,
        "architecture"    : "VGG16 (transfer learning)",
        "classes"         : CLASS_NAMES,
        "input_size"      : "256x256 RGB",
        "normalization"   : "divide by 255",
        "trainable_layers": "last 8 VGG16 layers",
        "class_weight"    : "balanced (imbalance fix applied)"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid file type '{file.content_type}'. Only JPEG and PNG allowed."
        )

    # Read file bytes
    contents = await file.read()

    # Validate file size
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code = 400,
            detail      = f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    # Run prediction
    try:
        predictor = BrainTumorPredictor.get_instance()
        result    = predictor.predict(contents)
        return result
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction failed: {str(e)}"
        )