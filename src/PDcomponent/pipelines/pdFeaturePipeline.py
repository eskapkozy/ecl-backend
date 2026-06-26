"""
PD Feature Pipeline
=====================
Spécialisation du FeaturePipeline pour le modèle de probabilité de défaut.

Spécifique au PD :
    - target binaire (défaut = 1 si max(DPD) >= 3 sur 12 mois)
    - transformation WoE
    - rééquilibrage SMOTE

Usage
-----
from pd_feature_pipeline import PDFeaturePipeline
from sklearn.model_selection import train_test_split

pipeline = PDFeaturePipeline(window_months=12, state="train")

# Etape 1 — features brutes (AVANT split)
X, y = pipeline.build(hist, orig, target="default")

# Etape 2 — split (hors pipeline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Etape 3 — WoE + SMOTE sur train uniquement
X_train_woe, y_train_bal = pipeline.apply_woe(X_train, y_train)

# Etape 4 — WoE sur test (réutilise le binning appris, pas de SMOTE)
X_test_woe, _ = pipeline.apply_woe(X_test)

# Inférence — nouveau prêt
pipeline_inf = PDFeaturePipeline(window_months=12, state="inference",
                                  binning_process=pipeline.binning_process)
pipeline_inf.selector = pipeline.selector
X_new        = pipeline_inf.build(hist_new, orig_new)
X_new_woe, _ = pipeline_inf.apply_woe(X_new)
"""

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.pipelines.Features.featurePipeline       import FeaturePipeline
from pipelines.Features.woe_pipeline import WoePipeline


class PDFeaturePipeline(FeaturePipeline):

    DEFAULT_WOE_CONFIG = {"iv_threshold": 0.02, "metric": "woe"}

    def __init__(self, window_months: int = 12, state: str = "train",
                 woe_config: dict = None, binning_process=None, config_path: dict = None):
        super().__init__(
            window_months=window_months,
            state=state,
            config_path=config_path,
        )
        self.woe_config       = {**self.DEFAULT_WOE_CONFIG, **(woe_config or {})}
        self.woe_pipeline_    = None
        self.binning_process  = binning_process
        self.target           = 'default'

    # ------------------------------------------------------------------
    # Target — binaire, 90 DPD, standard IFRS 9 / Bâle III
    # ------------------------------------------------------------------

    def _build_target(self, hist: pd.DataFrame) -> pd.DataFrame:
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
    # Rééquilibrage — spécifique classification
    # ------------------------------------------------------------------

    def _balance(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        smote = SMOTE()
        return smote.fit_resample(X, y)

    # ------------------------------------------------------------------
    # WoE + balance — APRÈS split, spécifique au scoring binaire
    # ------------------------------------------------------------------

    def apply_woe(self, X: pd.DataFrame, y: pd.Series = None) -> tuple:
        """
        y fourni (train)      : fit WoE + SMOTE, stocke le binning_process.
        y absent (test/infer) : réutilise le binning_process déjà appris.
        """
        if y is not None:
            self.woe_pipeline_   = WoePipeline(X, y, config=self.woe_config)
            self.binning_process = self.woe_pipeline_.capturedFit
            X_woe = self.woe_pipeline_.transform()
            X_woe, y = self._balance(X_woe, y)
            return X_woe, y                         # (X_woe, y)

        if self.binning_process is None:
            raise RuntimeError(
                "binning_process manquant — appelez apply_woe(X_train, y_train) d'abord."
            )

        woe_inf = WoePipeline(X, config=self.woe_config, binning_process=self.binning_process)
        return woe_inf.transform(), None # (X_woe, None)
