import mlflow
from loguru import logger


class MLflowTracker:

    def __init__(self, experiment_name: str = "rag-eval-framework", tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None
        logger.info(f"MLflow: experiment='{experiment_name}', uri='{tracking_uri}'")

    def start_run(self, run_name: str, params: dict = None) -> None:
        self._run = mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(params)
        logger.info(f"MLflow run started: {run_name}")

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def end_run(self) -> None:
        if self._run:
            mlflow.end_run()
            logger.info("MLflow run ended")
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end_run()
