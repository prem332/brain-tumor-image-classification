import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow : {tf.__version__}")
print(f"GPUs       : {tf.config.list_physical_devices('GPU')}")

# ── Config ──────────────────────────────────────────────────
BASE_PATH   = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset'
TRAIN_DATA  = os.path.join(BASE_PATH, 'Training')
TEST_DATA   = os.path.join(BASE_PATH, 'Testing')
IMAGE_SIZE  = (256, 256)
BATCH_SIZE  = 32
N_CLASSES   = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_SAVE  = '/kaggle/working/vgg_model_v3.h5'

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

# ── Data Generators ─────────────────────────────────────────
# FIX: Added brightness + contrast augmentation for better glioma features
train_datagen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 30,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    shear_range        = 0.2,
    zoom_range         = 0.3,
    horizontal_flip    = True,
    brightness_range   = [0.8, 1.2],   # NEW
    fill_mode          = 'nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DATA,
    target_size = IMAGE_SIZE,
    class_mode  = 'categorical',
    batch_size  = BATCH_SIZE,
    shuffle     = True
)
test_gen = test_datagen.flow_from_directory(
    TEST_DATA,
    target_size = IMAGE_SIZE,
    class_mode  = 'categorical',
    batch_size  = BATCH_SIZE,
    shuffle     = False
)

print(f"Class indices : {train_gen.class_indices}")
print(f"Train samples : {train_gen.samples}")
print(f"Test samples  : {test_gen.samples}")

# Class distribution
unique, counts = np.unique(train_gen.classes, return_counts=True)
print("\nClass distribution:")
for i, cnt in zip(unique, counts):
    print(f"  {CLASS_NAMES[i]:12s}: {cnt}")

# Class weights
y_train = train_gen.classes
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: float(cw[i]) for i in range(len(cw))}
print(f"\nClass weights : {class_weight_dict}")

# ── Build Model ─────────────────────────────────────────────
# FIX: GlobalAveragePooling2D instead of Flatten
# Flatten: 8*8*512 = 32768 -> 1024 = 33M params (overfits)
# GAP:     512 -> 1024 = 524K params (much leaner)

vgg = VGG16(
    input_shape = (256, 256, 3),
    include_top = False,
    weights     = 'imagenet'
)

# Freeze ALL VGG16 layers for Phase 1
for layer in vgg.layers:
    layer.trainable = False

# Custom head with GlobalAveragePooling
x = GlobalAveragePooling2D()(vgg.output)   # FIX: was Flatten
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(N_CLASSES, activation='softmax')(x)

model = Model(vgg.input, x)

# ── PHASE 1: Train head only ─────────────────────────────────
print("\n" + "="*50)
print("PHASE 1: Training head only (VGG16 frozen)")
print("="*50)

model.compile(
    optimizer = Adam(learning_rate=0.001),   # higher LR for head training
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

print(f"Trainable params phase 1: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

callbacks_phase1 = [
    EarlyStopping(
        monitor              = 'val_loss',
        patience             = 5,
        restore_best_weights = True,
        verbose              = 1
    ),
    ModelCheckpoint(
        filepath       = '/kaggle/working/checkpoints/phase1_best.h5',
        monitor        = 'val_accuracy',
        save_best_only = True,
        verbose        = 1
    )
]

history1 = model.fit(
    train_gen,
    epochs          = 15,
    validation_data = test_gen,
    class_weight    = class_weight_dict,
    callbacks       = callbacks_phase1,
    verbose         = 1
)

phase1_acc = max(history1.history['val_accuracy'])
print(f"\nPhase 1 best val_accuracy: {phase1_acc*100:.2f}%")

# ── PHASE 2: Fine-tune last 12 layers ───────────────────────
print("\n" + "="*50)
print("PHASE 2: Fine-tuning last 12 VGG16 layers")
print("="*50)

# Unfreeze last 12 layers (block4 full + block5 full)
for layer in vgg.layers:
    layer.trainable = False
for layer in vgg.layers[-12:]:
    layer.trainable = True

print("Trainable layers:")
for layer in vgg.layers:
    if layer.trainable:
        print(f"  {layer.name}")

# Recompile with very low LR to avoid destroying imagenet weights
model.compile(
    optimizer = Adam(learning_rate=0.00001),  # 100x lower than phase 1
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

print(f"Trainable params phase 2: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

callbacks_phase2 = [
    EarlyStopping(
        monitor              = 'val_loss',
        patience             = 7,
        restore_best_weights = True,
        verbose              = 1
    ),
    ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-7,
        verbose  = 1
    ),
    ModelCheckpoint(
        filepath       = '/kaggle/working/checkpoints/phase2_best.h5',
        monitor        = 'val_accuracy',
        save_best_only = True,
        verbose        = 1
    )
]

history2 = model.fit(
    train_gen,
    epochs          = 30,
    validation_data = test_gen,
    class_weight    = class_weight_dict,
    callbacks       = callbacks_phase2,
    verbose         = 1
)

phase2_acc = max(history2.history['val_accuracy'])
print(f"\nPhase 2 best val_accuracy: {phase2_acc*100:.2f}%")

# ── Save Model ───────────────────────────────────────────────
model.save(MODEL_SAVE)
print(f"\nModel saved: {MODEL_SAVE}")

model_config = {
    "version"           : "v3",
    "classes"           : CLASS_NAMES,
    "class_indices"     : train_gen.class_indices,
    "image_size"        : list(IMAGE_SIZE),
    "architecture"      : "VGG16 + GlobalAveragePooling2D",
    "normalization"     : "divide_by_255",
    "class_weight_used" : class_weight_dict,
    "trainable_layers"  : "last_12_vgg_layers",
    "training_phases"   : 2,
    "phase1_best_acc"   : round(phase1_acc * 100, 2),
    "phase2_best_acc"   : round(phase2_acc * 100, 2)
}
with open('/kaggle/working/model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
print("model_config.json saved")

# ── Combined Training Plots ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combine both phase histories
acc  = history1.history['accuracy']     + history2.history['accuracy']
val  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss']         + history2.history['loss']
vlos = history1.history['val_loss']     + history2.history['val_loss']
p1_end = len(history1.history['accuracy'])

axes[0].plot(acc,  label='Train Accuracy', color='blue')
axes[0].plot(val,  label='Val Accuracy',   color='orange')
axes[0].axvline(x=p1_end, color='gray', linestyle='--', label='Phase 1→2')
axes[0].set_title('Accuracy — Phase 1 + Phase 2')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss, label='Train Loss', color='blue')
axes[1].plot(vlos, label='Val Loss',   color='orange')
axes[1].axvline(x=p1_end, color='gray', linestyle='--', label='Phase 1→2')
axes[1].set_title('Loss — Phase 1 + Phase 2')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150)
plt.show()

# ── Final Evaluation ─────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\n{'='*50}")
print(f"FINAL TEST ACCURACY : {test_acc * 100:.2f}%")
print(f"FINAL TEST LOSS     : {test_loss:.4f}")
print(f"TARGET              : 90.00%")
print(f"STATUS              : {'✓ PASSED' if test_acc >= 0.90 else '✗ Below target'}")
print(f"{'='*50}")

y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
y_true = test_gen.classes

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot       = True,
    fmt         = 'd',
    cmap        = 'Blues',
    xticklabels = CLASS_NAMES,
    yticklabels = CLASS_NAMES
)
plt.title(f'Confusion Matrix — Accuracy: {test_acc*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
