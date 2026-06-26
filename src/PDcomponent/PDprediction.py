import pandas as pd

from PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from pipelines.Features.feature_selector import FeatureSelector
from src.predictionAbstraction import PredictionAbstraction


class PDPrediction(PredictionAbstraction):

    def __init__(self, mlflow_config: str, model_config: str, hist: pd.DataFrame, orig: pd.DataFrame):
        '''
        Args:
            mlflow_config (str): path to mlflow config
            model_config (str): path to model config
            hist (pd.DataFrame): historical data
            orig (pd.DataFrame): original data

            On admet que les donnees sont deja clean dans le warehouse et transmis selon une fenettre que le model a appris
        '''

        super().__init__(hist=hist, orig=orig, mlflow_config=mlflow_config, model_config=model_config)

    def setup(self):
        artifacts = self._load_data()

        binning_process, self._model_fit, _ = self._load_data()

        # build

        scaler = self._load_scaler()
        self._featurePipeline = PDFeaturePipeline(
            window_months=12,
            woe_config=self._model_config['woe'],
            binning_process=binning_process,
            state='prediction'
        )

        self._x_data = self._featurePipeline.build(self._hist,self._orig,scaler)









    def apply(self):
        x_woe, _ = self._featurePipeline.apply_woe(self._x_data)
        return self._model_fit.predict_proba(x_woe)[:, 1]