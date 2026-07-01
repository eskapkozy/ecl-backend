import pandas as pd



from LGDcomponent.pipelines.lgdFeaturePipeline import LGDFeaturePipeline
from src.predictionAbstraction import PredictionAbstraction


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
        self.bin_edges_ = self.discretizer.bin_edges_


        scaler = self._load_scaler()

        # scale x data

        self._featurePipeline = LGDFeaturePipeline(
            state= 'Prediction',
            n_bins=self._model_config['model']['n_bins']
        )

        self._x_data = self._featurePipeline.build(self._hist, self._orig, scaler)









    def _load_data(self):
        run_id = self._mlflow_config['run_id']

        config = [
            self._mlflow_config['lgd_discretizer'],
            self._mlflow_config['model_fit'],

        ]

        return self._artifactmanager.load_All(run_id=run_id, configList=config)