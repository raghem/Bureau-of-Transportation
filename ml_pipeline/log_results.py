import mlflow, json, os

artifact_dir = "./artifacts"

with open(os.path.join(artifact_dir, "results.json"), "r") as f:
    results = json.load(f)

mlflow.log_param("alpha", results["alpha"])
mlflow.log_param("order", results["order"])
mlflow.log_metric("mse", results["mse"])
mlflow.log_artifact(os.path.join(artifact_dir, "model.pkl"))


