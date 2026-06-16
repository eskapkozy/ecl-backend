import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum



import joblib
import json
import mlflow
import pandas as pd


class ArtifactManager:

    '''
        Le mode de tockage des artifact de mlflow est le suivant:

        artifacts/
            model_fit/        ← artifact_path="model_fit"
                model_fit.joblib
            binning_process/
                binning_process.joblib
  w         oe_iv_report/
                woe_iv_report.json


    '''





    # =========================
    # PUBLIC API
    # =========================
    def log(self, obj, name: str, artifact_type, ismodel=False):
        if artifact_type == ArtifactType.PKL:
            self._log_pkl(obj, name, ismodel=ismodel)
        elif artifact_type == ArtifactType.JSON:
            self._log_json(obj, name)
        elif artifact_type == ArtifactType.CSV:
            self._log_csv(obj, name)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")


    # =========================
    # PUBLIC API — LOAD
    # =========================

    def load_All(self, run_id: str, configList: list):

        def load_item(item):
            return self.load(
                run_id,
                item['name'],
                item['path'],
                item['artifact_type'],
                item['ismodel']
            )

        artifacts = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(load_item, item): i
                for i, item in enumerate(configList)
            }

            artifacts = [None] * len(configList)

            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    artifacts[i] = future.result()
                except Exception as e:
                    print("Artifact load failed:", e)

        return artifacts

    def load(self, run_id: str,path:str, name: str, artifact_type, ismodel=False):
        if ArtifactType[artifact_type] == ArtifactType.PKL:
            return self._load_pkl(run_id, path ,name, ismodel=ismodel)
        elif ArtifactType[artifact_type] == ArtifactType.JSON:
            return self._load_json(run_id,path , name)
        elif ArtifactType[artifact_type] == ArtifactType.CSV:
            return self._load_csv(run_id, path,name)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type} \n s'aaurer que les type dans config.yaml sont corrects identique a [PKL, JSON, CSV]")





    # =========================
    # PKL / JOBLIB
    # =========================
    def _log_pkl(self, obj, name: str, ismodel=False):
        if ismodel:
            mlflow.sklearn.log_model(obj, artifact_path=name)
            return
        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.joblib")
        joblib.dump(obj, tmp_path)
        try:
            mlflow.log_artifact(tmp_path, artifact_path=name)
        finally:
            os.remove(tmp_path)

    def _load_pkl(self, run_id: str, item_path:str,name: str, ismodel=False):

        relative_path = item_path.split("/artifacts/")[-1] if "/artifacts/" in str(item_path) else item_path

        if ismodel:
            return mlflow.sklearn.load_model(f"runs:/{run_id}/{name}")
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"{relative_path}/{name}.joblib"
        )
        return joblib.load(path)




    # =========================
    # JSON
    # =========================
    def _log_json(self, obj, name: str):

        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.json")

        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=4, default=str)

        try:
            mlflow.log_artifact(tmp_path, artifact_path=name)
        finally:
            os.remove(tmp_path)

    def _load_json(self, run_id: str, item_path,name: str) -> dict:

        relative_path = item_path.split("/artifacts/")[-1] if "/artifacts/" in str(item_path) else item_path

        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"{relative_path}/{name}.json"
        )
        with open(path, "r") as f:
            return json.load(f)




    # =========================
    # CSV
    # =========================
    def _log_csv(self, obj, name: str):

        if not isinstance(obj, pd.DataFrame):
            raise TypeError("CSV artifact must be a pandas DataFrame")

        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.csv")

        obj.to_csv(tmp_path, index=False)

        try:
            mlflow.log_artifact(tmp_path, artifact_path=name)
        finally:
            os.remove(tmp_path)

    def _load_csv(self, run_id: str, item_path,name: str) -> pd.DataFrame:

        relative_path = item_path.split("/artifacts/")[-1] if "/artifacts/" in str(item_path) else item_path

        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"{relative_path}/{name}.csv"
        )
        return pd.read_csv(path)





    # =========================
    # WOE binning table
    # =========================
    def log_woeT0_json(self, obj, name: str):

        def serialize(o):
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient="records")

            elif isinstance(o, dict):
                return {k: serialize(v) for k, v in o.items()}

            elif isinstance(o, list):
                return [serialize(i) for i in o]

            else:
                return o

        safe_obj = serialize(obj)

        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.json")

        with open(tmp_path, "w") as f:
            json.dump(safe_obj, f, indent=4, default=str)

        try:
            mlflow.log_artifact(tmp_path, artifact_path=name)
        finally:
            os.remove(tmp_path)


    def load_woe_table(self, run_id: str, item_path,name: str) -> dict:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"{item_path}/{name}.json"
        )
        with open(path, "r") as f:
            raw = json.load(f)
        return {k: pd.DataFrame(v) for k, v in raw.items()}



# =========================
# ENUM FIX (IMPORTANT CORRECTION)
# =========================
class ArtifactType(Enum):
    PKL = "pkl"
    JSON = "json"
    CSV = "csv"