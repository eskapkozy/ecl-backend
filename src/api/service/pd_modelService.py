# services/pd_model_service.py
import os
import pandas as pd
from src.api.warehouse.warehouseLoader import WarehouseReader
from src.PDcomponent.PDprediction import PDPrediction


class PDModelService:

    def __init__(self):
        self._reader = WarehouseReader(db_url=os.environ["WAREHOUSE_URL"])

    def predict(self, loan_id: str) -> float:
        hist, orig = self._reader.fetch(loan_id)
        prediction = PDPrediction(
            hist=hist,
            orig=orig,
            mlflow_config=os.environ["MLFLOW_CONFIG_PATH"],
            model_config=os.environ["MODEL_CONFIG_PATH"]
        )
        return float(prediction.apply()[0])