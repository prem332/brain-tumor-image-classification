# GCP Settings
GCP_PROJECT  = "brain-tumor-ai-prod"
GCS_BUCKET   = "brain-tumor-ai-models"
REGION       = "us-central1"

# Model Settings
MODEL_GCS_PATH  = "models/v3/vgg_model.h5"
CONFIG_GCS_PATH = "models/v3/model_config.json"
MODEL_VERSION   = "v3"
MODEL_LOCAL_PATH = "/tmp/vgg_model.h5"

# Image Settings
IMAGE_SIZE     = (256, 256)
IMAGE_CHANNELS = 3

# Class labels — order matches train_gen.class_indices
# {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Human-readable info per class
CLASS_INFO = {
    "glioma": {
        "label"      : "Glioma",
        "description": "A tumor that originates in the glial cells of the brain or spine",
        "severity"   : "high"
    },
    "meningioma": {
        "label"      : "Meningioma",
        "description": "A tumor that forms on the membranes surrounding the brain and spinal cord",
        "severity"   : "medium"
    },
    "notumor": {
        "label"      : "No Tumor",
        "description": "No tumor detected in this MRI scan",
        "severity"   : "none"
    },
    "pituitary": {
        "label"      : "Pituitary Tumor",
        "description": "A tumor that develops in the pituitary gland at the base of the brain",
        "severity"   : "medium"
    }
}

# API Settings
MAX_FILE_SIZE_MB = 10
ALLOWED_TYPES    = ["image/jpeg", "image/png", "image/jpg"]
API_VERSION      = "v1"