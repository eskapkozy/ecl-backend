"""
LGD Feature Pipeline
======================
Pas de WindowBuilder : on sélectionne d'abord les loans en défaut
(EAD identifié), puis on construit les features sur leur historique
complet — aucune troncature 12 mois, ni sur la target ni sur les features.
"""
from venv import logger

import pandas as pd
import numpy as np
import yaml

from LGDcomponent.lossFeature import LossFeatures
import src.LGDcomponent.lgd_Functions as lgd_func
from src.pipelines.Features.lgd_geo import GeoFeatures
from src.pipelines.Features.feature_selector import FeatureSelector
from src.pipelines.Features.featurePipeline import FeaturePipeline
from src.pipelines.Features.delinquency_features import DelinquencyFeatures
from src.pipelines.Features.capital_features import CapitalFeatures
from src.pipelines.Features.origination_features import OriginationFeatures
from src.pipelines.Features.Lgd_discretizer import LGDDiscretizer


RESOLUTION_CODES = {2, 3, 9, 15}
DEFAULT_DPD_THRESHOLD = 3


class LGDFeaturePipeline(FeaturePipeline):

    def __init__(self, state: str = "train", n_bins: int = 8, config_path: str = None):

        #config = self.get_config(config_path) if config_path is not None else None

        super().__init__(window_months=None, state=state, config_path=config_path)
        self.target = "lgd_target"
        self.discretizer = LGDDiscretizer(n_bins=n_bins)



    # ------------------------------------------------------------------
    # Sélection des loans en défaut — sur hist complet
    # ------------------------------------------------------------------

    @staticmethod
    def _select_defaulted(hist: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre hist sur les loans ayant atteint DPD >= 3 au moins une fois.
        Retourne l'historique complet (toutes lignes) de ces loans uniquement —
        pas de troncature temporelle.
        """




        df = hist.copy()
        df["CURRENT_LOAN_DELINQUENCY_STATUS"] = pd.to_numeric(
            df["CURRENT_LOAN_DELINQUENCY_STATUS"], errors="coerce")

        defaulted_ids = (
            df.loc[df["CURRENT_LOAN_DELINQUENCY_STATUS"] >= DEFAULT_DPD_THRESHOLD,
                   "LOAN_SEQUENCE_NUMBER"]
            .unique()
        )
        return df[df["LOAN_SEQUENCE_NUMBER"].isin(defaulted_ids)]

    # ------------------------------------------------------------------
    # Target — EAD vs recovery réel, sur hist complet du loan en défaut
    # ------------------------------------------------------------------

    def _build_target(self, hist_defaulted: pd.DataFrame) -> pd.DataFrame:
        return lgd_func.compute_lgd_target(hist_defaulted)

    # ------------------------------------------------------------------
    # Imputation — spécifique LGD, override du parent
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._impute(df)  # garde recuperation, freq_x_recuperation, progression, anticipation

        # LossFeatures
        df["n_months_in_default"] = df["n_months_in_default"].fillna(0)
        df["time_to_max_dpd"] = df["time_to_max_dpd"].fillna(0)
        df["ltv_at_default"] = df["ltv_at_default"].fillna(df["ltv_at_default"].median())

        # CapitalFeatures — ecart_au_plan peut être NaN si rate=0 et n=0 (edge case)
        df["ecart_au_plan"] = df["ecart_au_plan"].fillna(0)

        # DelinquencyFeatures — tendance NaN si un seul point (den=0)
        df["tendance"] = df["tendance"].fillna(0)

        # OriginationFeatures — couverture_mi déjà fillna(0) en amont, ok
        # GeoFeatures — property_type/property_state : pas de NaN attendu (table origination propre)

        return df
    # ------------------------------------------------------------------
    # Features — pas de WindowBuilder, historique complet des défauts
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Features — pas de WindowBuilder, historique complet des défauts
    # ------------------------------------------------------------------

    def _build_features(self, hist: pd.DataFrame, orig: pd.DataFrame) -> tuple:
        hist_defaulted = hist.copy()

        f_delinquency = DelinquencyFeatures(hist_defaulted).build()
        f_capital = CapitalFeatures(hist_defaulted, orig_df=orig).build()
        f_origination = OriginationFeatures(orig).build()
        f_loss = LossFeatures(hist_defaulted).build()
        f_geo = GeoFeatures(orig).build()  # ← ajout

        data = (
            f_delinquency
            .merge(f_capital, on="LOAN_SEQUENCE_NUMBER", how="inner")
            .merge(f_origination, on="LOAN_SEQUENCE_NUMBER", how="inner")
            .merge(f_loss, on="LOAN_SEQUENCE_NUMBER", how="inner")
            .merge(f_geo, on="LOAN_SEQUENCE_NUMBER", how="inner")  # ← ajout
        )

        return self._impute(data), hist

    # ------------------------------------------------------------------
    # build() — target sur hist_defaulted,
    # ------------------------------------------------------------------

    def build(self, hist: pd.DataFrame, orig: pd.DataFrame, scaler: dict = None):
        data, hist_defaulted = self._build_features(hist, orig)

        if self.state == "train":
            target_df = self._build_target(hist_defaulted).loc[:,['LOAN_SEQUENCE_NUMBER','lgd_target']]
            data = data.merge(target_df, on="LOAN_SEQUENCE_NUMBER", how="inner")
            x, y = self.selector.fit_transform(data, target=self.target)


            self.scaler_run_id = self.selector.scaler_run_id



            logger.info('build success')
            return x, y

        selector = FeatureSelector()
        return selector.transform(data, scaler)

    def apply_discretization(self, y: pd.Series = None, proba: np.ndarray = None):
        if y is not None:
            return self.discretizer.fit_transform(y)
        if proba is not None:
            return self.discretizer.expected_value(proba)
        raise ValueError("Fournir soit y (train), soit proba (inférence).")

    def get_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)