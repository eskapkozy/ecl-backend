"""
Feature Engineering — Groupe Perte (LossFeatures)
====================================================
Spécifique LGD — features absentes de DelinquencyFeatures/CapitalFeatures
car sans intérêt pour PD : durée du défaut, vitesse de dégradation,
equity au moment du défaut.

Hypothèses :
    - df déjà restreint aux loans en défaut (LossFeatures._select_defaulted
      en amont, dans LGDFeaturePipeline)
    - df trié chronologiquement par loan (MONTHLY_REPORTING_PERIOD croissant)
"""

import pandas as pd
import numpy as np


class LossFeatures:

    LOAN_COL = "LOAN_SEQUENCE_NUMBER"
    DPD_COL  = "CURRENT_LOAN_DELINQUENCY_STATUS"
    DATE_COL = "MONTHLY_REPORTING_PERIOD"
    LTV_COL  = "ESTIMATED_LTV"

    DPD_THRESHOLD = 3

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.df[self.DATE_COL] = pd.to_datetime(self.df[self.DATE_COL])
        self.df[self.DPD_COL] = pd.to_numeric(self.df[self.DPD_COL], errors="coerce").fillna(0)
        self.df[self.LTV_COL] = pd.to_numeric(self.df[self.LTV_COL], errors="coerce")

        self.df = self.df.sort_values([self.LOAN_COL, self.DATE_COL])

        self.df["_t"] = self.df.groupby(self.LOAN_COL).cumcount()
        self.df["_en_defaut"] = (self.df[self.DPD_COL] >= self.DPD_THRESHOLD).astype(int)

        self.gpr = self.df.groupby(self.LOAN_COL)

    # ------------------------------------------------------------------
    # Durée du défaut — proxy de coût de workout
    # ------------------------------------------------------------------

    def _n_months_in_default(self) -> pd.Series:
        return self.gpr["_en_defaut"].sum().rename("n_months_in_default")

    # ------------------------------------------------------------------
    # Vitesse de dégradation — temps entre t0 et le pic de DPD
    # ------------------------------------------------------------------

    def _time_to_max_dpd(self) -> pd.Series:
        max_dpd = self.gpr[self.DPD_COL].transform("max")
        at_max = (self.df[self.DPD_COL] == max_dpd) & (max_dpd > 0)

        # premier t où le pic est atteint
        t_at_max = self.df["_t"].where(at_max)
        t_first_default = (
            self.df["_t"].where(self.df["_en_defaut"] == 1)
            .groupby(self.df[self.LOAN_COL]).transform("min")
        )

        delta = (t_at_max - t_first_default).groupby(self.df[self.LOAN_COL]).min()
        return delta.rename("time_to_max_dpd")

    # ------------------------------------------------------------------
    # Equity au moment du défaut — LTV au premier mois DPD >= 3
    # ------------------------------------------------------------------

    def _ltv_at_default(self) -> pd.Series:
        first_default = (
            self.df[self.df["_en_defaut"] == 1]
            .groupby(self.LOAN_COL)
            .first()
        )
        return first_default[self.LTV_COL].rename("ltv_at_default")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        n_months   = self._n_months_in_default().reset_index()
        time_to_max= self._time_to_max_dpd().reset_index()
        ltv        = self._ltv_at_default().reset_index()

        return (
            n_months
            .merge(time_to_max, on=self.LOAN_COL, how="left")
            .merge(ltv, on=self.LOAN_COL, how="left")
        )