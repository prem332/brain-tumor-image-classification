#!/bin/bash
echo "Triggering manual Cloud Build..."
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=brain-tumor-ai-prod
echo "Done"
