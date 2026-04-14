import numpy as np
import io
import os
from PIL import Image
from tensorflow.keras.models import load_model
from google.cloud import storage
from config import (
    GCS_BUCKET, MODEL_GCS_PATH, MODEL_LOCAL_PATH,
    IMAGE_SIZE, CLASS_NAMES, CLASS_INFO, MODEL_VERSION
)

class BrainTumorPredictor:
    _instance = None
    model     = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print(f"[Predictor] Downloading model from GCS...")
        print(f"[Predictor] Bucket : {GCS_BUCKET}")
        print(f"[Predictor] Path   : {MODEL_GCS_PATH}")

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(MODEL_GCS_PATH)
        blob.download_to_filename(MODEL_LOCAL_PATH)

        print(f"[Predictor] Model downloaded to {MODEL_LOCAL_PATH}")

        self.model = load_model(MODEL_LOCAL_PATH)
        print(f"[Predictor] Model loaded successfully — version {MODEL_VERSION}")
        print(f"[Predictor] Input shape  : {self.model.input_shape}")
        print(f"[Predictor] Output shape : {self.model.output_shape}")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize to 256x256 — same as training
        img = img.resize(IMAGE_SIZE)

        # Convert to array and normalize — same as training (divide by 255)
        arr = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        arr   = self.preprocess(image_bytes)
        probs = self.model.predict(arr, verbose=0)[0]   # shape (4,)

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        # All class scores as percentages
        all_scores = {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class" : pred_class,
            "label"           : CLASS_INFO[pred_class]["label"],
            "confidence"      : round(confidence * 100, 2),
            "severity"        : CLASS_INFO[pred_class]["severity"],
            "description"     : CLASS_INFO[pred_class]["description"],
            "all_scores"      : all_scores,
            "model_version"   : MODEL_VERSION
        }
