import pandas as pd



from src.LGDcomponent.pipelines.lgdFeaturePipeline import LGDFeaturePipeline
from src.predictionAbstraction import PredictionAbstraction

import logging

logger = logging.getLogger(__name__)

class LGDPrediction(PredictionAbstraction):

    def __init__(self,
                 hist: pd.DataFrame,
                 orig: pd.DataFrame,
                 mlflow_config: str = None,
                 model_config: str = None):
        super().__init__(hist, orig, mlflow_config, model_config)

        self.discretizer = None
        self.bin_edges_ = None
        self.proba = None

        self.setup()



    def lgd(self, proba):
        estimated_lgd = []

        for pb in proba:
            midpoints = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
            compute = sum(pb * midpoints)
            estimated_lgd.append(compute)

        return estimated_lgd


    def apply(self):
        self.proba = self._model_fit.predict_proba(self._x_data)

        return self.lgd(self.proba)





    def setup(self):

        # load data


        self.discretizer, self._model_fit = self._load_data()

        self.load_data_error_log()




        self.bin_edges_ = self.discretizer.bin_edges_


        scaler = self._load_scaler()

        self.scaler_error_log(scaler)



        logger.info("Building feature pipeline...")

        # scale x data

        self._featurePipeline = LGDFeaturePipeline(
            state= 'Prediction',
            n_bins=self._model_config['model']['n_bins']
        )

        logger.info("Feature matrix built successfully.")

        self._x_data = self._featurePipeline.build(self._hist, self._orig, scaler)









    def _load_data(self):

        logger.info("Loading prediction artifacts...")
        run_id = self._mlflow_config['run_id']

        config = [
            self._mlflow_config['lgd_discretizer'],
            self._mlflow_config['model_fit'],

        ]

        return self._artifactmanager.load_All(run_id=run_id, configList=config)


    def load_data_error_log(self):

        if self.discretizer is None:
            logger.error("Discretizer artifact is None.")
            raise RuntimeError("Failed to load discretizer artifact.")

        logger.info("Discretizer loaded successfully.")

        if self._model_fit is None:
            logger.error("Model artifact is None.")
            raise RuntimeError("Failed to load model artifact.")

        logger.info("Prediction model loaded successfully.")

    def scaler_error_log(self, scaler):

        if scaler is None:
            logger.error("Scaler artifact is None.")
            raise RuntimeError("Failed to load scaler artifact.")

        logger.info("Scaler loaded successfully.")