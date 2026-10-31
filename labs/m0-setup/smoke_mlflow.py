
"""Smoke test MLflow : logge un run (paramètre, métriques, artefact) sur le serveur."""

import os
import tempfile

import mlflow

# Serveur MLflow du cluster, exposé sur le poste par make pf.
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
)
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello-mlops") as run:
    mlflow.log_param("framework", "scikit-learn")
    mlflow.log_metric("accuracy", 0.42)
    mlflow.log_metric("loss", 1.337)

    # L'artefact passe par le serveur MLflow jusqu'au bucket S3.
    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = os.path.join(tmp, "hello.txt")
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write("Hello MLOps!")

        mlflow.log_artifact(artifact_path)

    print("Run logged successfully.")
    print(f"run_id : {run.info.run_id}")
    print(f"artifact_uri : {run.info.artifact_uri}")
