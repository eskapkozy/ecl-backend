from abc import abstractmethod

import mlflow
import yaml
import pandas as pd
from optbinning.binning import binning_process

from PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from pipelines.Features import featurePipeline
from pipelines.Features.cleaningPipeline import CleaningPipeline
from src.PDcomponent.pipelines.pdFeaturePipeline import FeaturePipeline
from src.Utile.artifactManager import ArtifactManager



class PredictionAbstraction:
    def __init__(self, mlflow_config: dict,model_config: dict, loan_number:str,hist_path: str, orig_path: str ):

        self._model_config = self.load_config(model_config)

        self._hist_data, self._orig_data = CleaningPipeline(
            loan_number=loan_number,hist_path=hist_path, orig_path=orig_path).apply()




        self._featurePipeline = None

        self._x_data = None


        self._mlflow_config = self.load_config(mlflow_config)
        self._artifactmanager = ArtifactManager()




        mlflow.set_tracking_uri(self._mlflow_config['tracking_uri']) #self._mlflow_config['tracking_uri']

        mlflow.set_experiment(self._mlflow_config['experiment_name'])









    @abstractmethod
    def apply(self):
        raise NotImplementedError



    def load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)


    def setup(self, type: str, binning_process,scaler: dict):

        if type == 'PD':

            self._featurePipeline = PDFeaturePipeline(
            window_months=12,
            woe_config=self._model_config['woe'],
            binning_process=binning_process,
            state='prediction')

            self._x_data = self._featurePipeline.build(hist= self._hist_data, orig= self._orig_data,scaler=scaler)






    def load_model(self):

        def _reconstruct_model( ensemble, stacking_weights):
            ensemble.stacking_weights_ = stacking_weights

            return ensemble


        def _load_data():
            run_id = self._mlflow_config['run_id']

            config = [
                self._mlflow_config['binning_process'],
                self._mlflow_config['model_fit'],
                self._mlflow_config['stacking_weights'],
            ]

            #artifacts = self._artifactmanager.load_All(run_id=run_id, configList=config)

            #binning_process = artifacts[0]
            #model_fit = _reconstruct_model(artifacts[1], artifacts[2])

            return self._artifactmanager.load_All(run_id=run_id, configList=config)

        return  _load_data()


    def _load_data(self):
        run_id = self._mlflow_config['run_id']

        config = [
            self._mlflow_config['binning_process'],
            self._mlflow_config['model_fit'],
            self._mlflow_config['stacking_weights'],
            self._mlflow_config['scaler']
        ]

        return self._artifactmanager.load_All(run_id=run_id, configList=config)
