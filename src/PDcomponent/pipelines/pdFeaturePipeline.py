"""
PD Feature Pipeline
=====================
Spécialisation du FeaturePipeline pour le modèle de probabilité de défaut.

Spécifique au PD :
    - target binaire — trois couches de lecture du défaut (voir _build_target)
    - transformation WoE
    - rééquilibrage SMOTE

Définition du défaut — trois couches (IFRS 9 / Basel III) :

    Couche 1 — Signal structurel (sans condition DPD) :
        ZERO_BALANCE_CODE ∈ {2, 3, 9, 15, 16, 96} dans la window.
        Ces codes indiquent une résolution forcée (foreclosure, short sale,
        REO, note sale, relinquishment, deed-in-lieu) — le défaut est
        acquis indépendamment du comportement de paiement observé.

    Couche 2 — Signal comportemental :
        DPD >= 3 (90 jours de retard) — seuil réglementaire Basel III.

    Couche 3 — Qualification du signal DPD (persistance / cure) :
        Sur les loans actifs (ZERO_BALANCE_CODE = 0), DPD >= 3 est
        confirmé comme défaut uniquement si le loan présente au moins
        3 mois consécutifs à DPD >= 3 dans la window SANS cure
        subséquent (3 mois consécutifs à DPD = 0 après le run).
        Standard Basel III sur fenêtre 12 mois.

    default = signal_structurel | (signal_dpd & persistance & ~cure)

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
import numpy as np
from imblearn.over_sampling import SMOTE

from src.pipelines.Features.featurePipeline import FeaturePipeline
from src.pipelines.Features.woe_pipeline import WoePipeline


# Codes ZERO_BALANCE indiquant un défaut structurel — résolution forcée
STRUCTURAL_DEFAULT_CODES = {2, 3, 9, 15, 16, 96}

# Seuil réglementaire DPD (Basel III)
DPD_THRESHOLD = 3

# Nombre de mois consécutifs requis pour confirmer persistance ou cure
PERSISTENCE_MONTHS = 3
CURE_MONTHS = 3


class PDFeaturePipeline(FeaturePipeline):

    DEFAULT_WOE_CONFIG = {"iv_threshold": 0.02, "metric": "woe"}

    def __init__(self, window_months: int = 12, state: str = "train",
                 woe_config: dict = None, binning_process=None, config_path: str = None):
        super().__init__(
            window_months=window_months,
            state=state,
            config_path=config_path,
        )
        self.woe_config      = {**self.DEFAULT_WOE_CONFIG, **(woe_config or {})}
        self.woe_pipeline_   = None
        self.binning_process = binning_process
        self.target          = 'default'

    # ------------------------------------------------------------------
    # Couche 1 — Signal structurel
    # ------------------------------------------------------------------

    @staticmethod
    def _structural_default(hist: pd.DataFrame) -> pd.Series:
        """
        Retourne une Series booléenne indexée sur LOAN_SEQUENCE_NUMBER.
        True si ZERO_BALANCE_CODE ∈ STRUCTURAL_DEFAULT_CODES dans la window.
        Ces codes constituent un défaut sans condition DPD.
        """
        zbc = pd.to_numeric(hist["ZERO_BALANCE_CODE"], errors="coerce")
        flag = zbc.isin(STRUCTURAL_DEFAULT_CODES)
        return (
            flag.groupby(hist["LOAN_SEQUENCE_NUMBER"])
            .any()
        )

    # ------------------------------------------------------------------
    # Couche 2+3 — Signal comportemental + persistance / cure
    # ------------------------------------------------------------------

    @staticmethod
    def _max_consecutive(series: pd.Series, condition: pd.Series) -> int:
        """
        Calcule le nombre maximum de valeurs consécutives satisfaisant
        une condition booléenne dans une Series ordonnée.
        """
        count = 0
        max_count = 0
        for val in condition[series.index]:
            if val:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    @staticmethod
    def _has_persistence(dpd_series: pd.Series) -> bool:
        """
        Vérifie si le loan présente au moins PERSISTENCE_MONTHS mois
        consécutifs à DPD >= DPD_THRESHOLD dans la window.
        Standard Basel III sur fenêtre 12 mois.
        """
        in_default = dpd_series >= DPD_THRESHOLD
        count = 0
        for val in in_default:
            count = count + 1 if val else 0
            if count >= PERSISTENCE_MONTHS:
                return True
        return False

    @staticmethod
    def _has_cure(dpd_series: pd.Series) -> bool:
        """
        Vérifie si le loan présente un cure après défaut :
        au moins CURE_MONTHS mois consécutifs à DPD = 0
        après avoir atteint DPD >= DPD_THRESHOLD.
        Un cure annule le signal comportemental.
        """
        in_default = False
        cure_count = 0

        for dpd in dpd_series:
            if dpd >= DPD_THRESHOLD:
                in_default = True
                cure_count = 0
            elif in_default:
                cure_count += 1
                if cure_count >= CURE_MONTHS:
                    return True
        return False

    @staticmethod
    def _behavioral_default(hist: pd.DataFrame) -> pd.Series:
        """
        Retourne une Series booléenne indexée sur LOAN_SEQUENCE_NUMBER.
        True si le loan satisfait persistance confirmée SANS cure subséquent.
        Appliqué uniquement sur les loans actifs (ZERO_BALANCE_CODE = 0).

        Implémentation vectorisée — pas de boucle Python par loan.
        Logique : un "run" de valeurs consécutives satisfaisant une condition
        peut être détecté avec le pattern groupby-cumsum classique :
            group_id = (~condition).cumsum()
            run_length = condition.groupby([loan_id, group_id]).cumsum()
        """
        dpd = pd.to_numeric(
            hist["CURRENT_LOAN_DELINQUENCY_STATUS"].replace("RA", -1),
            errors="coerce"
        ).fillna(0)

        zbc = pd.to_numeric(hist["ZERO_BALANCE_CODE"], errors="coerce").fillna(0)

        df = pd.DataFrame({
            "loan_id": hist["LOAN_SEQUENCE_NUMBER"].values,
            "dpd": dpd.values,
            "zbc": zbc.values,
        })

        # Ordre chronologique requis pour la logique de run — supposé déjà trié
        # par MONTHLY_REPORTING_PERIOD en amont (WindowBuilder).

        in_default_flag = df["dpd"] >= DPD_THRESHOLD

        # --- Run de persistance : longueur de séquence consécutive en défaut ---
        # group_id incrémente à chaque rupture de la condition (False)
        persistence_break = (~in_default_flag).groupby(df["loan_id"]).cumsum()
        persistence_run = (
            in_default_flag
            .groupby([df["loan_id"], persistence_break])
            .cumsum()
        )
        has_persistence = (
            (persistence_run >= PERSISTENCE_MONTHS)
            .groupby(df["loan_id"])
            .any()
        )

        # --- Run de cure : longueur de séquence consécutive à DPD=0, ---
        # ---  uniquement après qu'un défaut ait été atteint au moins une fois ---
        ever_defaulted = (
            in_default_flag
            .groupby(df["loan_id"])
            .cummax()
        )
        cure_candidate = (~in_default_flag) & ever_defaulted

        cure_break = (~cure_candidate).groupby(df["loan_id"]).cumsum()
        cure_run = (
            cure_candidate
            .groupby([df["loan_id"], cure_break])
            .cumsum()
        )
        has_cure = (
            (cure_run >= CURE_MONTHS)
            .groupby(df["loan_id"])
            .any()
        )

        # Défaut comportemental = persistance confirmée AND pas de cure
        behavioral = has_persistence & ~has_cure

        # Restreindre aux loans actifs (ZERO_BALANCE_CODE = 0)
        is_active_loan = (df["zbc"] == 0).groupby(df["loan_id"]).all()

        result = behavioral & is_active_loan

        all_loans = hist["LOAN_SEQUENCE_NUMBER"].unique()
        return result.reindex(all_loans, fill_value=False)

    # ------------------------------------------------------------------
    # Target — orchestration des trois couches
    # ------------------------------------------------------------------

    def _build_target(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Construit la target binaire en combinant les trois couches.

        default = signal_structurel | signal_comportemental
        """
        structural  = self._structural_default(hist)
        behavioral  = self._behavioral_default(hist)

        default = (structural | behavioral).astype(int)

        return (
            default
            .reset_index()
            .rename(columns={0: "default"})
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
            return X_woe, y

        if self.binning_process is None:
            raise RuntimeError(
                "binning_process manquant — appelez apply_woe(X_train, y_train) d'abord."
            )

        woe_inf = WoePipeline(X, config=self.woe_config, binning_process=self.binning_process)
        return woe_inf.transform(), None