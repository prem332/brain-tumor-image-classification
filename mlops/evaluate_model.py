import numpy as np
import json
import os
from google.cloud import storage
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

GCP_PROJECT  = "brain-tumor-ai-prod"
GCS_BUCKET   = "brain-tumor-ai-models"
CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE   = (256, 256)
BATCH_SIZE   = 32
THRESHOLD    = 0.80

def download_model(version: str = "v2") -> str:
    print(f"[Evaluate] Downloading model {version} from GCS...")
    client   = storage.Client(project=GCP_PROJECT)
    bucket   = client.bucket(GCS_BUCKET)
    blob     = bucket.blob(f"models/{version}/vgg_model.h5")
    local    = f"/tmp/vgg_model_{version}.h5"
    blob.download_to_filename(local)
    print(f"[Evaluate] Model downloaded to {local}")
    return local

def evaluate(test_data_path: str, version: str = "v2"):
    # Load model
    model_path = download_model(version)
    model      = tf.keras.models.load_model(model_path)
    print(f"[Evaluate] Model loaded")

    # Load test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen     = test_datagen.flow_from_directory(
        test_data_path,
        target_size = IMAGE_SIZE,
        class_mode  = 'categorical',
        batch_size  = BATCH_SIZE,
        shuffle     = False
    )

    # Evaluate
    print(f"[Evaluate] Running evaluation on {test_gen.samples} test images...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    print(f"\n{'='*50}")
    print(f"Model Version : {version}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Threshold     : {THRESHOLD * 100:.2f}%")
    print(f"{'='*50}")

    # Per-class report
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    y_true = test_gen.classes

    print("\nPer-Class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Deployment decision
    if test_acc >= THRESHOLD:
        print(f"\n[Evaluate] PASS — Model accuracy {test_acc*100:.2f}% meets threshold. Safe to deploy.")
    else:
        print(f"\n[Evaluate] FAIL — Model accuracy {test_acc*100:.2f}% below threshold {THRESHOLD*100:.2f}%. Do NOT deploy.")

    # Save results to GCS
    results = {
        "version"      : version,
        "test_loss"    : round(test_loss, 4),
        "test_accuracy": round(test_acc * 100, 2),
        "threshold"    : THRESHOLD * 100,
        "passed"       : bool(test_acc >= THRESHOLD),
        "classes"      : CLASS_NAMES
    }
    results_path = "/tmp/eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(f"models/{version}/eval_results.json")
    blob.upload_from_filename(results_path)
    print(f"[Evaluate] Results saved to gs://{GCS_BUCKET}/models/{version}/eval_results.json")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True, help="Path to test data folder")
    parser.add_argument("--version",   default="v2",  help="Model version e.g. v2")
    args = parser.parse_args()
    evaluate(args.test_data, args.version)