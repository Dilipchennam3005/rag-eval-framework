import os
import wandb
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class WandbTracker:

    def __init__(self, project: str = "rag-eval-framework"):
        self.project = project
        self._run = None

    def start_run(self, run_name: str, config: dict = None) -> None:
        self._run = wandb.init(
            project=self.project,
            name=run_name,
            config=config or {},
            reinit=True,
        )
        logger.info(f"W&B run started: {run_name} (project={self.project})")

    def log(self, metrics: dict, step: int = None) -> None:
        if self._run:
            wandb.log(metrics, step=step)

    def log_table(self, name: str, data: list[dict]) -> None:
        if self._run and data:
            import pandas as pd
            table = wandb.Table(dataframe=pd.DataFrame(data))
            wandb.log({name: table})

    def end_run(self) -> None:
        if self._run:
            self._run.finish()
            logger.info("W&B run finished")
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end_run()
