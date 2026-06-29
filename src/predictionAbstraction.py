import os
from abc import abstractmethod, ABC

import mlflow
import yaml
import pandas as pd

from src.Utile.artifactManager import ArtifactManager


class PredictionAbstraction(ABC):

    def __init__(self,
                 hist: pd.DataFrame,
                 orig: pd.DataFrame,
                 mlflow_config: str = None,
                 model_config: str = None):

        mlflow_config = mlflow_config or os.environ["MLFLOW_CONFIG_PATH"]
        model_config  = model_config  or os.environ["MODEL_CONFIG_PATH"]

        self._model_config  = self.load_config(model_config)
        self._mlflow_config = self.load_config(mlflow_config)
        self._artifactmanager = ArtifactManager()

        mlflow.set_tracking_uri(self._mlflow_config['tracking_uri'])
        mlflow.set_experiment(self._mlflow_config['experiment_name'])

        self._hist   = hist
        self._orig   = orig
        self._x_data = None
        self._featurePipeline = None

        self.setup()

    @abstractmethod
    def apply(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    def load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_data(self):
        run_id = self._mlflow_config['run_id']
        config = [
            self._mlflow_config['binning_process'],
            self._mlflow_config['model_fit'],
            #self._mlflow_config['stacking_weights'],
        ]
        return self._artifactmanager.load_All(run_id=run_id, configList=config)

    def _load_scaler(self) -> dict:
        scaler_config = self._mlflow_config['preprocessing']['scaler']
        return self._artifactmanager.load(
            run_id=scaler_config['run_id'],
            path=scaler_config['name'],
            name=scaler_config['name'],
            artifact_type="PKL",
            ismodel=False
        )