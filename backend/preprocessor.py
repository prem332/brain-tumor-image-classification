import numpy as np
from PIL import Image
import io
from config import IMAGE_SIZE

def validate_image(image_bytes: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_image_info(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    return {
        "format"        : img.format,
        "original_size" : img.size,
        "mode"          : img.mode,
        "resized_to"    : IMAGE_SIZE
    }
