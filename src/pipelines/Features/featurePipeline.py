"""
Feature Pipeline — Classe de base
====================================
Responsabilité : construction des features communes à tous les modèles
                 ECL (PD, LGD, EAD) — fenêtre, retard, capital, origination.

Cette classe est abstraite : _build_target() et la transformation finale
(apply_woe / apply_scaling) doivent être implémentées par les sous-classes.

Voir architecture_feature_pipeline.md pour la stratégie complète.

Usage
-----
class PDFeaturePipeline(FeaturePipeline):
    def _build_target(self, hist): ...
    def apply_woe(self, X, y=None): ...
"""

from abc import ABC, abstractmethod
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from pipelines.Features.window_builder        import WindowBuilder
from pipelines.Features.delinquency_features  import DelinquencyFeatures
from pipelines.Features.capital_features      import CapitalFeatures
from pipelines.Features.origination_features  import OriginationFeatures
from pipelines.Features.feature_selector      import FeatureSelector


class FeaturePipeline(ABC):

    def __init__(self, window_months: int = 12, state: str = "train"):
        self.window_months = window_months
        self.selector       = FeatureSelector()
        self.state          = state
        self.target         = None

        self.scaler_artifact = None
    # ------------------------------------------------------------------
    # Construction de la target — spécifique à chaque modèle
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_target(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Doit retourner un DataFrame avec LOAN_SEQUENCE_NUMBER + colonne target.

        PD  : target binaire (défaut/non-défaut)
        LGD : target continue (% de perte)
        EAD : target continue (montant d'exposition)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Imputation — commune
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["recuperation"]        = df["recuperation"].fillna(0)
        df["freq_x_recuperation"] = df["freq_x_recuperation"].fillna(0)
        df["progression"]         = df["progression"].fillna(0)
        df["anticipation"]        = df["anticipation"].fillna(df["anticipation"].median())
        return df

    # ------------------------------------------------------------------
    # Construction des features — commune à tous les modèles
    # ------------------------------------------------------------------

    def _build_features(self, hist: pd.DataFrame, orig: pd.DataFrame) -> tuple:
        """
        Retourne (data, hist_12m).
        Identique pour PD, LGD, EAD — fenêtre + retard + capital + origination.
        """
        n_loans = hist["LOAN_SEQUENCE_NUMBER"].nunique()
        if n_loans == 1:
            hist_12m = (
                hist.sort_values("MONTHLY_REPORTING_PERIOD")
                .tail(self.window_months)
            )
        else:
            hist_12m = WindowBuilder(hist, window_months=self.window_months).build()

        with ThreadPoolExecutor(max_workers=3) as executor:
            f_delinquency = executor.submit(lambda: DelinquencyFeatures(hist_12m.copy()).build())
            f_capital     = executor.submit(lambda: CapitalFeatures(hist_12m.copy(), orig_df=orig).build())
            f_origination = executor.submit(lambda: OriginationFeatures(orig).build())

        delinquency_agg = f_delinquency.result()
        capital_agg     = f_capital.result()
        orig_agg        = f_origination.result()

        data = (
            delinquency_agg
            .merge(capital_agg, on="LOAN_SEQUENCE_NUMBER", how="inner")
            .merge(orig_agg,    on="LOAN_SEQUENCE_NUMBER", how="inner")
        )
        data = self._impute(data)

        return data, hist_12m

    # ------------------------------------------------------------------
    # build() — features brutes scalées, AVANT split. Commun.
    # ------------------------------------------------------------------

    def build(self, hist: pd.DataFrame, orig: pd.DataFrame,scaler: dict = None):
        """
        Mode train      : retourne (X, y)
        Mode inference   : retourne X seul
        Ne déclenche jamais la transformation finale (WoE/scaling) —
        celle-ci doit être appliquée après le split, via une méthode
        propre à chaque sous-classe (apply_woe, apply_scaling, ...).
        """
        data, hist_12m = self._build_features(hist, orig)

        if self.state == "train":
            target_df = self._build_target(hist_12m)
            data      = data.merge(target_df, on="LOAN_SEQUENCE_NUMBER", how="inner")

            x,y = self.selector.fit_transform(data, target=self.target)

            self.scaler_artifact = self.selector.artifact
            return x,y




        return self.selector.transform(data,scaler)