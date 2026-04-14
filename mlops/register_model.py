from google.cloud import aiplatform
import argparse

GCP_PROJECT = "brain-tumor-ai-prod"
REGION      = "us-central1"
GCS_BUCKET  = "brain-tumor-ai-models"

def register_model(version: str = "v2"):
    print(f"[Registry] Initializing Vertex AI...")
    aiplatform.init(project=GCP_PROJECT, location=REGION)

    artifact_uri = f"gs://{GCS_BUCKET}/models/{version}/"

    print(f"[Registry] Registering model version {version}...")
    print(f"[Registry] Artifact URI: {artifact_uri}")

    model = aiplatform.Model.upload(
        display_name         = f"brain-tumor-vgg16-{version}",
        artifact_uri         = artifact_uri,
        serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
        description          = f"VGG16 4-class brain tumor classifier — {version} (class_weight fix applied)",
        labels               = {
            "version"      : version,
            "architecture" : "vgg16",
            "framework"    : "tensorflow",
            "task"         : "image-classification",
            "classes"      : "4"
        }
    )

    print(f"[Registry] Model registered successfully")
    print(f"[Registry] Model name    : {model.display_name}")
    print(f"[Registry] Model version : {model.version_id}")
    print(f"[Registry] Resource name : {model.resource_name}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", help="Model version e.g. v2")
    args = parser.parse_args()
    register_model(args.version)