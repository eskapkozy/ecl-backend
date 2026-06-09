from src.api.utils import load_config
import os


class Mlflow_config:

    def __init__(self):
        config_path = os.getenv("MLFLOW_CONFIG_PATH", "configs/mlflow_config.yaml")

        self.config = load_config(config_path)

        self.run_id = self.config['run_id']
        self.tracking_uri = self.config['tracking_uri']
        self.experiment_name = self.config['experiment_name']

        self.binning_process = self.config['binning_process']
        self.model_fit = self.config['model_fit']
        self.stacking_weights = self.config['stacking_weights']

        # todo : etendre la config pour les autre models
        # todo : plusieur autre config , donc evoluer par focntion mais en declarons les attribut au prealable