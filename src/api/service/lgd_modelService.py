import os

from src.api.warehouse.warehouseLoader import WarehouseReader
from src.LGDcomponent.LgdPrediction import LGDPrediction


class LgdModelService:

    def __init__(self):
        self._reader = WarehouseReader(db_url=os.environ["WAREHOUSE_URL"])

    def predict(self, loan_id: str) -> float:
        hist, orig = self._reader.fetch(loan_id)
        prediction = LGDPrediction(
            hist=hist,
            orig=orig,
            mlflow_config=os.environ["LGD_MLFLOW_CONFIG_PATH"],
            model_config=os.environ["LGD_MODEL_CONFIG_PATH"]
        )
        return float(prediction.apply()[0])