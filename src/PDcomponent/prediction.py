import mlflow
import pandas as pd

from predictionAbstraction import PredictionAbstraction
from src.PDcomponent.pipelines.pdFeaturePipeline import  PDFeaturePipeline
from src.Utile.artifactManager import ArtifactManager



class Prediction(PredictionAbstraction):
    def __init__(self, mlflow_config: dict,model_config: dict,loan_number:str, hist_path: str, orig_path: str):
        super().__init__(mlflow_config, model_config,loan_number ,hist_path,orig_path)



    # todo: add logging
    def apply(self):
        # ########################
        # Load Artifact
        # ########################

        binning_process, model_fit ,_,scaler= self._load_data()

        # ########################
        # Test data transformation
        # ########################

        self.setup(type='PD', binning_process=binning_process,scaler=scaler)
        x_transformed, _ = self._featurePipeline.apply_woe(self._x_data)




        # #############
        # Prediction from x_transformed
        # #############

        y_prob = model_fit.predict_proba(x_transformed)[:, 1]



        return y_prob








