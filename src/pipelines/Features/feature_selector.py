"""
Feature Selector
=================
Responsabilité : sélection et mise à l'échelle des features
                 basées sur les décisions du notebook EDA.

Optimisation : 2 appels sklearn au lieu de 19 (un par groupe de scaling).
"""

import pandas as pd
import logging
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.Utile.artifactManager import ArtifactManager, ArtifactType


class FeatureSelector:

    TO_DROP = [
        "produit_risque",
        "severite",
        "recidivisme_extreme",
        "freq_x_recuperation",
        "freq_x_tendance",
        "progression",
    ]

    SCALING = {
        "mensualite_implicite"  : "robust",
        "charge_taux_duree"     : "robust",
        "pression_levier"       : "robust",
        "ecart_ltv_ocltv"       : "robust",
        "couverture_mi"         : "robust",
        "multi_unite"           : None,
        "occupancy_risk"        : None,
        "refi_flag"             : None,
        "credit_segment"        : None,
        "primo_accedant"        : None,
        "co_emprunteur"         : None,
        "freq"                  : "standard",
        "tendance"              : "robust",
        "recuperation"          : "robust",
        "profondeur_max"        : "robust",
        "n_profondeur_max"      : "robust",
        "freq_x_profondeur_max" : "robust",
        "niveau"                : "standard",
        "ecart_au_plan"         : "robust",
        "anticipation"          : "standard",
    }

    def __init__(self,  scaler_logging_config: dict = None):
        self.features_        = list(self.SCALING.keys())
        self.robust_cols_     = [f for f, m in self.SCALING.items() if m == "robust"]
        self.standard_cols_   = [f for f, m in self.SCALING.items() if m == "standard"]
        self.robust_scaler_   = RobustScaler()
        self.standard_scaler_ = StandardScaler()

        self.artifact = None
        self.artifacts = None
        self.scaler_run_id = None
        self.artifact_manager = ArtifactManager()
        self.scaler_logging_config = scaler_logging_config or {}



    # ------------------------------------------------------------------
    # Fit + Transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, target: str) -> tuple:




        y = df[target].copy()
        X = df[self.features_].copy()


        #
        X[self.robust_cols_]   = self.robust_scaler_.fit_transform(X[self.robust_cols_])
        X[self.standard_cols_] = self.standard_scaler_.fit_transform(X[self.standard_cols_])


        # capture des artefacts du fit
        self.artifacts = {
            "robust_scaler": self.robust_scaler_,
            "standard_scaler": self.standard_scaler_,
            "features": self.features_,
            "robust_cols": self.robust_cols_,
            "standard_cols": self.standard_cols_,
            "scaling": self.SCALING,
        }


        self._log_scaler_artifact()


        return X, y

    def configure_scaler_logging(self, scaler_logging_config: dict = None):

        self.scaler_logging_config = scaler_logging_config



    def _log_scaler_artifact(self):
        config = self.scaler_logging_config or {}
        if not config.get("enabled", False):
            return


        run_name = config.get("run_name", "PD_scaler_preprocessing")
        nested = mlflow.active_run() is not None

        if config.get("tracking_uri"):
            mlflow.set_tracking_uri(config["tracking_uri"])
        if config.get("experiment_name"):
            mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            self.scaler_run_id = run.info.run_id
            mlflow.set_tags({
                "run_role": config.get("run_role", "preprocessing"),
                "artifact_kind": "scaler",
                "model_family": config.get("model_family", "PD"),
            })

            self.artifact_manager.log(
                obj=self.artifacts,
                name=config.get("name", "scaler"),
                artifact_type=ArtifactType.PKL,
            )

            self.artifact_manager.log(
                obj={
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "model_family": config.get("model_family", "PD"),
                    "features": self.features_,
                    "robust_cols": self.robust_cols_,
                    "standard_cols": self.standard_cols_,
                    "scaling": self.SCALING,
                },
                name=config.get("metadata_name", "scaler_metadata"),
                artifact_type=ArtifactType.JSON,
            )

    def transform(self, df: pd.DataFrame,scaler: dict) -> pd.DataFrame:


        self._setScaler(scaler)

        X = df[self.features_].copy()

        X[self.robust_cols_]   = self.robust_scaler_.transform(X[self.robust_cols_])
        X[self.standard_cols_] = self.standard_scaler_.transform(X[self.standard_cols_])

        # todo faire un load de l'artifavte du scaler
        return X

    def _setScaler(self, scaler: dict):
        self.robust_scaler_ = scaler["robust_scaler"]
        self.standard_scaler_ = scaler["standard_scaler"]
        self.features_ = scaler["features"]

    # ------------------------------------------------------------------
    # Rapport
    # ------------------------------------------------------------------

    def report(self) -> pd.DataFrame:
        rows = []
        for f, method in self.SCALING.items():
            rows.append({"feature": f, "statut": "retenu", "scaling": method or "aucun"})
        for f in self.TO_DROP:
            rows.append({"feature": f, "statut": "évincer", "scaling": "-"})
        return pd.DataFrame(rows)
