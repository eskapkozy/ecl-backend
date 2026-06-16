"""
Feature Pipeline
=================
Responsabilité : orchestration de toutes les étapes de feature engineering.

Prend en entrée :
    - hist     : DataFrame historique mensuel Freddie Mac
    - orig     : DataFrame origination Freddie Mac
    - target   : nom de la colonne cible (défaut = "default")

Retourne :
    - X        : DataFrame features prêt pour l'entraînement
    - y        : Series target

Usage
-----
from feature_pipeline import FeaturePipeline

pipeline   = FeaturePipeline(window_months=12)
X, y       = pipeline.fit_transform(hist, orig)

# Inférence
X_new      = pipeline.transform(hist_new, orig_new)
"""

import pandas as pd
import numpy as np
from optbinning import OptimalBinning

from src.pipelines.window_builder           import WindowBuilder
from src.pipelines.delinquency_features     import DelinquencyFeatures
from src.pipelines.capital_features         import CapitalFeatures
from src.pipelines.origination_features     import OriginationFeatures
from src.pipelines.feature_selector         import FeatureSelector
from src.pipelines.woe_pipeline             import WoePipeline

from concurrent.futures import ThreadPoolExecutor


class FeaturePipeline:

    def __init__(self, window_months: int = 12, woe_config: dict = None):
        self.window_months = window_months
        self.selector      = FeatureSelector()
        self.woe_config    = woe_config or {"iv_threshold": 0.02, "metric": "woe"}
        self.woe_pipeline_ = None

    # ------------------------------------------------------------------
    # Construction de la target
    # ------------------------------------------------------------------

    def _build_target(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Target binaire : défaut = 1 si max(DPD) >= 3 dans l'outcome window.
        Seuil 3 = 90 DPD — standard IFRS 9 / Bâle III.
        """
        dpd = pd.to_numeric(
            hist["CURRENT_LOAN_DELINQUENCY_STATUS"].replace("RA", -1),
            errors="coerce"
        )
        return (
            dpd.groupby(hist["LOAN_SEQUENCE_NUMBER"])
            .max()
            .ge(3)
            .astype(int)
            .reset_index()
            .rename(columns={"CURRENT_LOAN_DELINQUENCY_STATUS": "default"})
        )

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["recuperation"]        = df["recuperation"].fillna(0)
        df["freq_x_recuperation"] = df["freq_x_recuperation"].fillna(0)
        df["progression"]         = df["progression"].fillna(0)
        df["anticipation"]        = df["anticipation"].fillna(df["anticipation"].median())
        return df

    # ------------------------------------------------------------------
    # Transformation commune (fit et inférence)
    # ------------------------------------------------------------------

    def _build_features(self, hist: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
        # Etape 1 — Fenêtre
        # Mode batch     : plusieurs prêts → WindowBuilder
        # Mode inférence : un seul prêt   → tail(window_months) directement
        n_loans = hist["LOAN_SEQUENCE_NUMBER"].nunique()
        if n_loans == 1:
            hist_12m = (
                hist
                .sort_values("MONTHLY_REPORTING_PERIOD")
                .tail(self.window_months)
            )
        else:
            hist_12m = WindowBuilder(hist, window_months=self.window_months).build()



        with ThreadPoolExecutor(max_workers=3) as executor:
            f_delinquency = executor.submit(
                lambda: DelinquencyFeatures(hist_12m.copy()).build()
            )
            f_capital = executor.submit(
                lambda: CapitalFeatures(hist_12m.copy(), orig_df=orig).build()
            )
            f_origination = executor.submit(
                lambda: OriginationFeatures(orig).build()
            )





        # Etape 2 — Features Retard
        delinquency_agg = f_delinquency.result()

        # Etape 3 — Features Capital
        capital_agg =  f_capital.result()

        # Etape 4 — Features Origination
        orig_agg = f_origination.result()

        # Etape 5 — Jointure
        data = (
            delinquency_agg
            .merge(capital_agg, on="LOAN_SEQUENCE_NUMBER", how="inner")
            .merge(orig_agg,    on="LOAN_SEQUENCE_NUMBER", how="inner")
        )

        # Etape 6 — Imputation
        data = self._impute(data)

        return data, hist_12m  # retourne les deux

        return data

    # ------------------------------------------------------------------
    # Fit Transform
    # ------------------------------------------------------------------

    def fit_transform(self, hist: pd.DataFrame,
                      orig: pd.DataFrame,
                      target: str = "default") -> tuple:
        """
        Construit les features, la target, applique le scaling.
        Retourne (X, y) prêts pour l'entraînement.
        """
        data, hist_12m = self._build_features(hist, orig)

        # Target sur la fenêtre d'observation
        target_df = self._build_target(hist_12m)
        data = data.merge(target_df, on="LOAN_SEQUENCE_NUMBER", how="inner")

        # Sélection + scaling
        X, y = self.selector.fit_transform(data, target=target)

        # WoE transform
        self.woe_pipeline_ = WoePipeline(X, y, config=self.woe_config)
        X_woe = self.woe_pipeline_.transform()

        return X_woe, y

    # --------------------------------
    # INtegration de variable
    # --------------------------------




    # ------------------------------------------------------------------
    # Transform (inférence)
    # ------------------------------------------------------------------

    def transform(self, hist: pd.DataFrame,
                  orig: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le pipeline sur de nouvelles données sans recalculer le scaling.
        """
        data, hist_12m = self._build_features(hist, orig)
        X = self.selector.transform(data)

        if self.woe_pipeline_ is None:
            raise RuntimeError("fit_transform doit être appelé avant transform.")

        woe_inf = WoePipeline(
            X,
            config=self.woe_config,
            binning_process=self.woe_pipeline_.capturedFit
        )
        return woe_inf.transform()