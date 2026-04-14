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
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
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
EPOCHS      = 30
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_SAVE  = '/kaggle/working/vgg_model_v2.h5'

print(f"\nTrain path : {TRAIN_DATA}")
print(f"Test path  : {TEST_DATA}")
print(f"Image size : {IMAGE_SIZE}")
print(f"Classes    : {CLASS_NAMES}")

# ── Data Generators ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale          = 1./255,
    rotation_range   = 30,
    width_shift_range= 0.2,
    height_shift_range=0.2,
    shear_range      = 0.2,
    zoom_range       = 0.3,
    horizontal_flip  = True
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

print(f"\nClass indices : {train_gen.class_indices}")
print(f"Train samples : {train_gen.samples}")
print(f"Test samples  : {test_gen.samples}")

# Class distribution
unique, counts = np.unique(train_gen.classes, return_counts=True)
print("\nClass distribution:")
for i, cnt in zip(unique, counts):
    print(f"  {CLASS_NAMES[i]:12s}: {cnt}")

# ── Class Weights (imbalance fix) ────────────────────────────
y_train = train_gen.classes
cw = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes      = np.unique(y_train),
    y            = y_train
)
class_weight_dict = {i: float(cw[i]) for i in range(len(cw))}
print(f"\nClass weights : {class_weight_dict}")

# ── Build Model ─────────────────────────────────────────────
vgg = VGG16(
    input_shape  = (256, 256, 3),
    include_top  = False,
    weights      = 'imagenet'
)

# Freeze all layers first
for layer in vgg.layers:
    layer.trainable = False

# Unfreeze last 8 layers (block5 full)
for layer in vgg.layers[-8:]:
    layer.trainable = True

print("\nTrainable layers:")
for layer in vgg.layers:
    if layer.trainable:
        print(f"  {layer.name}")

# Custom head — same as original
x = Flatten()(vgg.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(N_CLASSES, activation='softmax')(x)

model = Model(vgg.input, x)
model.compile(
    optimizer = Adam(learning_rate=0.00001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

print(f"\nTotal params     : {model.count_params():,}")

# ── Callbacks ───────────────────────────────────────────────
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor             = 'val_loss',
        patience            = 5,
        restore_best_weights= True,
        verbose             = 1
    ),
    ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-6,
        verbose  = 1
    ),
    ModelCheckpoint(
        filepath      = '/kaggle/working/checkpoints/best_model.h5',
        monitor       = 'val_accuracy',
        save_best_only= True,
        verbose       = 1
    )
]

# ── Train ────────────────────────────────────────────────────
print("\nStarting training...")
print("=" * 50)

history = model.fit(
    train_gen,
    epochs         = EPOCHS,
    validation_data= test_gen,
    class_weight   = class_weight_dict,
    callbacks      = callbacks,
    verbose        = 1
)

print("\nTraining complete!")

# ── Save Model ───────────────────────────────────────────────
model.save(MODEL_SAVE)
print(f"Model saved: {MODEL_SAVE}")

# Save config JSON
model_config = {
    "version"          : "v2",
    "classes"          : CLASS_NAMES,
    "class_indices"    : train_gen.class_indices,
    "image_size"       : list(IMAGE_SIZE),
    "architecture"     : "VGG16",
    "normalization"    : "divide_by_255",
    "class_weight_used": class_weight_dict,
    "trainable_layers" : "last_8_vgg_layers"
}
with open('/kaggle/working/model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
print("model_config.json saved")

# ── Plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='orange')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss', color='blue')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='orange')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150)
plt.show()

# ── Evaluate ─────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")

y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
y_true = test_gen.classes

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot        = True,
    fmt          = 'd',
    cmap         = 'Blues',
    xticklabels  = CLASS_NAMES,
    yticklabels  = CLASS_NAMES
)
plt.title(f'Confusion Matrix — Accuracy: {test_acc*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")