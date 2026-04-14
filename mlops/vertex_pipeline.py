from google.cloud import aiplatform
from google.cloud import storage
import json

GCP_PROJECT = "brain-tumor-ai-prod"
REGION      = "us-central1"
GCS_BUCKET  = "brain-tumor-ai-models"

def run_mlops_pipeline(version: str = "v2"):
    aiplatform.init(project=GCP_PROJECT, location=REGION)

    print(f"\n{'='*50}")
    print(f"Brain Tumor AI — MLOps Pipeline")
    print(f"Model version : {version}")
    print(f"{'='*50}\n")

    # Step 1: Read evaluation results from GCS
    print("[Pipeline] Step 1: Reading evaluation results from GCS...")
    client  = storage.Client(project=GCP_PROJECT)
    bucket  = client.bucket(GCS_BUCKET)
    blob    = bucket.blob(f"models/{version}/eval_results.json")

    results_path = "/tmp/eval_results.json"
    blob.download_to_filename(results_path)

    with open(results_path) as f:
        results = json.load(f)

    print(f"[Pipeline] Test Accuracy : {results['test_accuracy']}%")
    print(f"[Pipeline] Threshold     : {results['threshold']}%")
    print(f"[Pipeline] Passed        : {results['passed']}")

    # Step 2: Register model if evaluation passed
    if results["passed"]:
        print(f"\n[Pipeline] Step 2: Registering model in Vertex AI Model Registry...")

        model = aiplatform.Model.upload(
            display_name  = f"brain-tumor-vgg16-{version}",
            artifact_uri  = f"gs://{GCS_BUCKET}/models/{version}/",
            serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
            description   = f"VGG16 brain tumor classifier {version} — accuracy {results['test_accuracy']}%",
            labels        = {
                "version"  : version,
                "accuracy" : str(results["test_accuracy"]),
                "passed"   : "true"
            }
        )
        print(f"[Pipeline] Model registered: {model.display_name}")
        print(f"[Pipeline] Resource name  : {model.resource_name}")

        # Step 3: Log experiment metrics
        print(f"\n[Pipeline] Step 3: Logging metrics to Vertex AI Experiments...")
        experiment = aiplatform.Experiment.get_or_create(
            experiment_name = "brain-tumor-training",
            project         = GCP_PROJECT,
            location        = REGION
        )
        with aiplatform.start_run(run=f"run-{version}"):
            aiplatform.log_metrics({
                "test_accuracy": results["test_accuracy"],
                "test_loss"    : results["test_loss"],
                "threshold"    : results["threshold"]
            })
            aiplatform.log_params({
                "model_version"    : version,
                "architecture"     : "VGG16",
                "classes"          : "4",
                "class_weight_fix" : "applied",
                "trainable_layers" : "last_8"
            })
        print(f"[Pipeline] Metrics logged to Vertex AI Experiments")

    else:
        print(f"\n[Pipeline] Step 2: SKIPPED — model did not pass evaluation threshold")
        print(f"[Pipeline] Retrain with better hyperparameters before registering")

    print(f"\n[Pipeline] MLOps pipeline complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", help="Model version e.g. v2")
    args = parser.parse_args()
    run_mlops_pipeline(args.version)