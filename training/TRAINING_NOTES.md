# Brain Tumor AI — Kaggle Training Notes

## Platform
- Kaggle Notebooks (free tier)
- GPU: Tesla T4 x2 (13,757 MB each)
- TensorFlow: 2.19.0
- Python: 3.12

## Dataset
- Source: Kaggle — Brain Tumor MRI Dataset by masoudnickparvar
- Kaggle path: /kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Classes: glioma, meningioma, notumor, pituitary
- Train samples: 5,600 (1,400 per class — perfectly balanced)
- Test samples: 1,600 (400 per class)
- Google Drive backup: https://drive.google.com/drive/folders/1YZbI-G0cWrR50jMJhpAQTsxb_aWMdKGh

## Steps Performed in Kaggle

### Step 1: Clone training script from GitHub
    git clone https://github.com/prem332/brain-tumor-image-classification.git

### Step 2: Run v2 training (first attempt)
    python brain-tumor-image-classification/training/train.py

- Result: 84.25% test accuracy
- Issue: Glioma recall was only 0.61 — model missed 39% of glioma cases
- Root cause: Flatten layer caused overfitting (33M params), only 8 layers unfrozen

### Step 3: Pull updated script and run v3 training (two-phase)
    cd brain-tumor-image-classification && git pull
    python brain-tumor-image-classification/training/train.py

- Result: 94.44% test accuracy — target achieved

## Model Versions

### v2 — First Attempt
- Test Accuracy  : 84.25%
- Architecture   : VGG16 + Flatten
- Trainable layers: Last 8
- Class weight   : Applied
- Status         : Below 90% target — not deployed

### v3 — Production Model
- Test Accuracy  : 94.44%
- Architecture   : VGG16 + GlobalAveragePooling2D
- Training phases: 2 (head first, then fine-tune)
- Trainable layers: Last 12 (block3 + block4 + block5)
- Class weight   : Applied
- Status         : Deployed to production on Cloud Run

## Two-Phase Training Strategy (v3)

### Phase 1 — Train head only (VGG16 fully frozen)
- Learning rate  : 0.001 (high — safe since VGG16 frozen)
- Max epochs     : 15, early stopped at epoch 8
- Best val acc   : 80.69%
- Purpose        : Warm up custom head without disturbing ImageNet weights

### Phase 2 — Fine-tune last 12 VGG16 layers
- Learning rate  : 0.00001 (100x lower — protects ImageNet weights)
- Max epochs     : 30, early stopped at epoch 16 (restored epoch 9 weights)
- Best val acc   : 94.44%
- Key            : ModelCheckpoint saved best weights throughout training

## Per-Class Results (v3)

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Glioma     | 0.96      | 0.84   | 0.90 |
| Meningioma | 0.91      | 0.95   | 0.93 |
| No Tumor   | 0.93      | 1.00   | 0.96 |
| Pituitary  | 0.98      | 0.99   | 0.99 |
| Overall    | 0.95      | 0.94   | 0.94 |

## Key Bug Fixes from Original Code

| Bug                | Original                              | Fixed                                    |
|--------------------|---------------------------------------|------------------------------------------|
| Class weight       | Computed but never passed to fit()    | Passed as class_weight=class_weight_dict |
| Feature extraction | Flatten — 33M params, overfits        | GlobalAveragePooling2D — 524K params     |
| Unfrozen layers    | Last 8 only                           | Last 12 (full block3+block4+block5)      |
| Model saving       | Only final epoch saved                | ModelCheckpoint saves best val_accuracy  |
| Training strategy  | Single phase                          | Two-phase (head first, fine-tune second) |

## Model Artifacts in GCS

gs://brain-tumor-ai-models/models/v2/
  - vgg_model.h5              — First attempt (84.25%)

gs://brain-tumor-ai-models/models/v3/
  - vgg_model.h5              — Production model (94.44%)
  - model_config.json         — Class labels, image size, metadata
  - training_curves.png       — Accuracy and loss plots
  - confusion_matrix.png      — Per-class confusion matrix
  - savedmodel/               — SavedModel format for Vertex AI Registry
    - saved_model.pb
    - fingerprint.pb
    - variables/

## Vertex AI Model Registry
- Model name  : brain-tumor-vgg16-v3
- Version     : 1
- Resource    : projects/224944730814/locations/us-central1/models/2705043795421954048
- Artifact URI: gs://brain-tumor-ai-models/models/v3/savedmodel/