# Brain Tumor AI — GCP Production Deployment

A production-grade brain tumor classification system built on Google Cloud Platform. The system accepts MRI scan images and classifies them into four tumor categories using a fine-tuned VGG16 deep learning model, achieving **94.44% test accuracy**.

> **Note on Live URLs:** The live deployment URLs listed in this README may be temporarily suspended to avoid ongoing Google Cloud Platform billing charges. The project is fully functional and can be redeployed at any time using the CI/CD pipeline documented below. Production screenshots are available in the `app_results/` directory as proof of the working deployment.

---

## Live Deployment

| Service | URL |
|---|---|
| Frontend | https://brain-tumor-frontend-224944730814.us-central1.run.app |
| Backend API | https://brain-tumor-backend-224944730814.us-central1.run.app |
| API Docs | https://brain-tumor-backend-224944730814.us-central1.run.app/docs |

---

## Classification Categories

| Class | Severity | Description |
|---|---|---|
| Glioma | High | Tumor originating in the glial cells of the brain or spine |
| Meningioma | Medium | Tumor forming on the membranes surrounding the brain |
| Pituitary | Medium | Tumor developing in the pituitary gland |
| No Tumor | None | No tumor detected in the MRI scan |

---

## Tech Stack

### Machine Learning
- Model: VGG16 (Transfer Learning, TensorFlow 2.19.0)
- Training: Two-phase fine-tuning strategy
- Test Accuracy: 94.44% (4-class classification)
- Training Platform: Kaggle (Tesla T4 x2 GPU)

### Backend
- Framework: FastAPI
- Runtime: Python 3.12
- Model Storage: Google Cloud Storage
- Container: Docker (python:3.12-slim)

### Frontend
- Framework: Next.js 16 (TypeScript)
- Styling: Tailwind CSS + Inline styles
- Theme: Dark medical-grade UI
- Container: Docker (Node 20 Alpine, multi-stage build)

### Infrastructure
- Cloud Provider: Google Cloud Platform
- Deployment: Cloud Run (serverless containers)
- Container Registry: Artifact Registry
- Model Registry: Vertex AI Model Registry
- CI/CD: GitHub Actions + Cloud Build
- Storage: Google Cloud Storage

---

## Project Structure

    brain-tumor-ai/
    ├── backend/                   # FastAPI application
    │   ├── config.py              # Central configuration (GCS paths, class labels, thresholds)
    │   ├── main.py                # API endpoints (/predict /health /model-info)
    │   ├── predictor.py           # Model inference (singleton pattern)
    │   ├── preprocessor.py        # Image preprocessing utilities
    │   ├── requirements.txt       # Python dependencies
    │   └── Dockerfile             # Python 3.12-slim container
    │
    ├── frontend/                  # Next.js application
    │   ├── app/
    │   │   ├── page.tsx           # Main page with upload and prediction state
    │   │   ├── layout.tsx         # Root layout and metadata
    │   │   └── globals.css        # Global dark theme styles
    │   ├── components/
    │   │   ├── Header.tsx         # Sticky navigation header
    │   │   ├── UploadSection.tsx  # Drag-and-drop MRI image upload
    │   │   └── ResultCard.tsx     # Prediction result with confidence scores
    │   ├── .env.local             # Local development API URL
    │   ├── .env.production        # Production API URL
    │   └── Dockerfile             # Multi-stage Node 20 Alpine build
    │
    ├── mlops/                     # MLOps scripts
    │   ├── register_model.py      # Register model in Vertex AI Model Registry
    │   ├── evaluate_model.py      # Evaluate model against accuracy threshold
    │   └── vertex_pipeline.py     # Full MLOps pipeline orchestration
    │
    ├── training/                  # Model training
    │   ├── train.py               # Two-phase VGG16 training script (Kaggle)
    │   └── TRAINING_NOTES.md      # Complete Kaggle training documentation
    │
    ├── various_models_train/      # Original model comparison experiments
    │   ├── brain-tumor-image-classification9.ipynb
    │   └── all_models_results/    # Result plots from all model experiments
    │
    ├── data/
    │   └── DATA_SOURCES.md        # Kaggle and Google Drive dataset links
    │
    ├── app_results/               # Production screenshots (proof of deployment)
    │   └── result_1.png ...       # End-to-end prediction screenshots
    │
    ├── scripts/
    │   └── trigger_build.sh       # Manual Cloud Build trigger script
    │
    ├── .github/
    │   └── workflows/
    │       └── deploy.yml         # GitHub Actions CI/CD pipeline
    │
    ├── cloudbuild.yaml            # Cloud Build — builds and deploys both services
    ├── .gitignore
    └── README.md

---

## System Architecture

### High-Level Request Flow

    User (Browser)
         |
         | Upload MRI Image (.jpg / .png)
         v
    Next.js Frontend  ── Cloud Run (512Mi RAM)
         |
         | POST /predict  (multipart/form-data)
         v
    FastAPI Backend  ── Cloud Run (2Gi RAM, 2 CPU)
         |
         +──> 1. File type and size validation
         |
         +──> 2. Image preprocessing
         |         - Convert to RGB
         |         - Resize to 256x256
         |         - Normalize pixels (divide by 255)
         |         - Add batch dimension
         |
         +──> 3. VGG16 model inference
         |         - Softmax output: shape (4,)
         |         - argmax → predicted class
         |
         +──> 4. Build JSON response
         |
         v
    Next.js Frontend renders ResultCard
         |
         v
    User sees: class label, confidence %, severity badge,
               all class scores, model version

### How Prediction Works — Step by Step

**Step 1 — User Uploads MRI Image**
The user drags and drops or selects an MRI image in the Next.js frontend.
The frontend immediately previews the image and sends a POST request
to the FastAPI backend at /predict with the image as multipart form data.

**Step 2 — File Validation (FastAPI)**
The backend validates two things before touching the model:
- File type must be image/jpeg or image/png
- File size must not exceed 10MB

If either check fails, a 400 HTTP error is returned immediately
without loading or running the model.

**Step 3 — Image Preprocessing**
The image bytes are passed to the preprocessor which applies
exactly the same transformations used during training:
- Open image using Pillow
- Convert to RGB (handles grayscale MRI scans automatically)
- Resize to 256x256 pixels
- Convert pixel values to float32
- Normalize by dividing by 255.0 (range becomes 0.0 to 1.0)
- Add batch dimension: final shape is (1, 256, 256, 3)

**Step 4 — Model Inference**
The preprocessed tensor is passed to the VGG16 model which was
loaded from Google Cloud Storage at container startup. The model
outputs a softmax probability vector of shape (4,) representing
confidence scores for each of the four classes:
- Index 0: glioma
- Index 1: meningioma
- Index 2: notumor
- Index 3: pituitary

**Step 5 — Result Construction**
The backend identifies the class with the highest probability using
argmax, converts all probabilities to percentages, and builds a
structured JSON response including the predicted class, confidence
score, severity level, clinical description, and all class scores.

**Step 6 — Frontend Renders Result**
The Next.js frontend receives the JSON response and renders the
ResultCard component showing the prediction label, confidence bar,
severity badge (High / Medium / None), and per-class score breakdown
with color-coded progress bars.

---

## Model Internal Architecture

    Input: MRI Image (256 x 256 x 3)
         |
         v
    VGG16 Feature Extractor (pretrained on ImageNet)
    ├── Block 1: Conv(64)  → Conv(64)  → MaxPool    [FROZEN]
    ├── Block 2: Conv(128) → Conv(128) → MaxPool    [FROZEN]
    ├── Block 3: Conv(256) → Conv(256) → Conv(256) → MaxPool    [FINE-TUNED]
    ├── Block 4: Conv(512) → Conv(512) → Conv(512) → MaxPool    [FINE-TUNED]
    └── Block 5: Conv(512) → Conv(512) → Conv(512) → MaxPool    [FINE-TUNED]
         |
         v
    GlobalAveragePooling2D
    (reduces 8x8x512 feature maps to 512 — prevents overfitting)
         |
         v
    Dense(512, relu) → BatchNormalization → Dropout(0.5)
         |
         v
    Dense(256, relu) → BatchNormalization → Dropout(0.3)
         |
         v
    Dense(4, softmax)
         |
         v
    Output: [glioma, meningioma, notumor, pituitary] probabilities

### Model Loading Strategy — Singleton Pattern

The VGG16 model (~200MB) is loaded once when the Cloud Run container
starts, not on every request. This is implemented using the Singleton
design pattern in predictor.py:

- Container starts → BrainTumorPredictor.get_instance() called once
- Model downloaded from GCS to /tmp/vgg_model.h5
- Model loaded into memory and cached in the singleton instance
- Every subsequent /predict request reuses the cached model
- Cold start takes approximately 30-40 seconds (first request after idle)
- Warm requests complete in approximately 1-2 seconds

---

## Two-Phase Training Strategy

### Why Two Phases?

Training a transfer learning model in a single phase with a high
learning rate risks destroying the pre-trained ImageNet weights.
Training with a low learning rate from the start makes the custom
head converge too slowly. Two-phase training solves both problems.

### Phase 1 — Head Training (VGG16 fully frozen)
- All VGG16 layers are frozen (weights cannot change)
- Only the custom classification head is trainable (396K parameters)
- Learning rate: 0.001 (high — safe since VGG16 is frozen)
- Max epochs: 15, early stopped at epoch 8
- Best validation accuracy: 80.69%
- Purpose: Warm up the head to a reasonable starting point

### Phase 2 — Fine-Tuning (last 12 VGG16 layers unfrozen)
- Block 3, Block 4, and Block 5 of VGG16 become trainable (14.8M parameters)
- Learning rate: 0.00001 (100x lower — protects ImageNet weights)
- Max epochs: 30, early stopped at epoch 16
- EarlyStopping restored best weights from epoch 9
- Best validation accuracy: 94.44%

---

## Model Versions

### v2 — First Attempt (84.25% accuracy)
- Architecture: VGG16 + Flatten layer
- Flatten creates 33M parameters → significant overfitting
- Only last 8 VGG16 layers unfrozen
- Class weight computed but accidentally never passed to fit()
- Not deployed to production

### v3 — Production Model (94.44% accuracy)
- Architecture: VGG16 + GlobalAveragePooling2D
- GlobalAveragePooling reduces parameters to 524K → lean and robust
- Last 12 VGG16 layers unfrozen (block3 + block4 + block5)
- Class weight correctly passed to model.fit()
- Two-phase training strategy applied
- ModelCheckpoint saves best weights throughout training
- Deployed to production on Cloud Run

### Key Bug Fixes from Original Code

| Issue | Original Code | Fixed In v3 |
|---|---|---|
| Class weight | Computed but never passed to fit() | Passed as class_weight=class_weight_dict |
| Feature extraction | Flatten — 33M params, overfits | GlobalAveragePooling2D — 524K params |
| Unfrozen layers | Last 8 only | Last 12 (block3 + block4 + block5) |
| Model checkpoint | Only final epoch saved | ModelCheckpoint saves best val_accuracy |
| Training strategy | Single phase | Two-phase (head warm-up, then fine-tune) |

---

## Per-Class Performance (v3)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Glioma | 0.96 | 0.84 | 0.90 | 400 |
| Meningioma | 0.91 | 0.95 | 0.93 | 400 |
| No Tumor | 0.93 | 1.00 | 0.96 | 400 |
| Pituitary | 0.98 | 0.99 | 0.99 | 400 |
| Overall | 0.95 | 0.94 | 0.94 | 1600 |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Service info and status |
| GET | /health | Health check — model loaded status and version |
| GET | /model-info | Architecture details and version metadata |
| POST | /predict | Upload MRI image — returns classification result |

### Sample /predict Response

    {
      "predicted_class": "pituitary",
      "label": "Pituitary Tumor",
      "confidence": 99.54,
      "severity": "medium",
      "description": "A tumor that develops in the pituitary gland at the base of the brain",
      "all_scores": {
        "glioma": 0.01,
        "meningioma": 0.44,
        "notumor": 0.01,
        "pituitary": 99.54
      },
      "model_version": "v3"
    }

---

## CI/CD Pipeline

Every push to the main branch triggers the following pipeline automatically:

    GitHub push to main
         |
         v
    GitHub Actions (deploy.yml)
         |
         | gcloud builds submit
         v
    Cloud Build (cloudbuild.yaml)
         |
         +──> Build backend Docker image
         |         |
         |         v
         |    Push to Artifact Registry
         |         |
         |         v
         |    Deploy to Cloud Run (brain-tumor-backend)
         |         2Gi RAM, 2 CPU, port 8080
         |
         +──> Build frontend Docker image
                   |
                   v
              Push to Artifact Registry
                   |
                   v
              Deploy to Cloud Run (brain-tumor-frontend)
                   512Mi RAM, 1 CPU, port 3000

### GCP Infrastructure

    GitHub Repository
         |
    GitHub Actions ──> Cloud Build ──> Artifact Registry
                                            |
                              +─────────────+─────────────+
                              |                           |
                         Cloud Run                   Cloud Run
                         (Backend)                  (Frontend)
                              |
                         Google Cloud Storage
                    gs://brain-tumor-ai-models/
                              |
                    Vertex AI Model Registry
                    (versioning + lineage)

---

## Dataset

- Source: Brain Tumor MRI Dataset by masoudnickparvar
- Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Google Drive Backup: https://drive.google.com/drive/folders/1YZbI-G0cWrR50jMJhpAQTsxb_aWMdKGh
- Total images: 7,200 (5,600 training / 1,600 testing)
- Class balance: Perfectly balanced — 1,400 training images per class
- Data augmentation: rotation, shift, shear, zoom, horizontal flip, brightness

---

## GCS Model Storage

    gs://brain-tumor-ai-models/
    ├── models/v2/
    │   └── vgg_model.h5                   # First attempt — 84.25% accuracy
    └── models/v3/
        ├── vgg_model.h5                   # Production model — 94.44% accuracy
        ├── model_config.json              # Class labels, image size, metadata
        ├── training_curves.png            # Accuracy and loss plots
        ├── confusion_matrix.png           # Per-class confusion matrix
        └── savedmodel/                    # SavedModel format for Vertex AI
            ├── saved_model.pb
            ├── fingerprint.pb
            └── variables/

---

## Vertex AI Model Registry

- Model name: brain-tumor-vgg16-v3
- Version: 1
- Resource: projects/224944730814/locations/us-central1/models/2705043795421954048
- Artifact URI: gs://brain-tumor-ai-models/models/v3/savedmodel/

---

## Local Development

### Prerequisites
- Python 3.12
- Node.js 20
- Google Cloud SDK
- Docker

### Backend

    cd backend
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

### Frontend

    cd frontend
    npm install
    npm run dev

---

## Redeployment

To redeploy from scratch after a billing pause:

    gcloud config set project brain-tumor-ai-prod
    gcloud builds submit --config=cloudbuild.yaml --project=brain-tumor-ai-prod

---

## Disclaimer

This application is intended for research and portfolio demonstration purposes only. It is not a certified medical device and should not be used for clinical diagnosis or medical decision-making. Always consult a qualified healthcare professional for medical advice.

---

## Author

Prem Kumar — AI/ML Engineer
GitHub: https://github.com/prem332
Project Repository: https://github.com/prem332/brain-tumor-image-classification