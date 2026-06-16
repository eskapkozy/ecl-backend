"""
Feature Engineering — Groupe Capital (vectorisé v2)
=====================================================
Optimisations v2 :
    - first/last calculés une seule fois dans __init__
    - _progression : suppression du apply — diff vectorisé puis std
    - _anticipation : suppression de la copie inutile
"""

import pandas as pd
import numpy as np


class CapitalFeatures:

    LOAN_COL     = "LOAN_SEQUENCE_NUMBER"
    UPB_COL      = "CURRENT_ACTUAL_UPB"
    RATE_COL     = "CURRENT_INTEREST_RATE"
    AGE_COL      = "LOAN_AGE"
    REMAIN_COL   = "REMAINING_MONTHS_TO_LEGAL_MATURITY"
    ORIG_UPB_COL = "ORIGINAL_UPB"

    def __init__(self, df: pd.DataFrame, orig_df: pd.DataFrame):
        orig_upb = orig_df[["LOAN_SEQUENCE_NUMBER", "ORIGINAL_UPB"]]
        self.df  = df.merge(orig_upb, on="LOAN_SEQUENCE_NUMBER", how="left")

        for col in [self.UPB_COL, self.RATE_COL, self.AGE_COL,
                    self.REMAIN_COL, self.ORIG_UPB_COL]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # first / last calculés une seule fois pour toutes les méthodes
        self.grp        = self.df.groupby(self.LOAN_COL)
        self.first = self.grp.first().reset_index()
        self.last  = self.grp.last().reset_index()

        # diff UPB vectorisé une seule fois
        self.df["_diff_upb"] = self.grp[self.UPB_COL].diff()

    # Angle 1 — Niveau
    def _niveau(self) -> pd.Series:
        last = self.last[self.UPB_COL]
        orig = self.first[self.ORIG_UPB_COL].replace(0, np.nan)
        return (
            pd.Series(last.values / orig.values,
                      index=self.last[self.LOAN_COL],
                      name="niveau")
        )

    # Angle 2 — Progression (vectorisé — sans apply)
    def _progression(self) -> pd.Series:
        return (
            self.grp["_diff_upb"]
            .std()
            .rename("progression")
        )

    # Angle 3 — Ecart au plan
    def _ecart_au_plan(self) -> pd.Series:
        upb_orig = self.last[self.ORIG_UPB_COL]
        age      = self.last[self.AGE_COL]
        remain   = self.last[self.REMAIN_COL]
        rate     = self.last[self.RATE_COL]
        upb_reel = self.last[self.UPB_COL]

        n  = age + remain
        r  = rate / 100 / 12
        rn = (1 + r) ** n
        rt = (1 + r) ** age

        upb_th_zero = upb_orig * (1 - age / n.replace(0, np.nan))
        upb_th_rate = upb_orig * (rn - rt) / (rn - 1)
        upb_th      = np.where(r == 0, upb_th_zero, upb_th_rate)

        return pd.Series(
            upb_th - upb_reel.values,
            index=self.last[self.LOAN_COL],
            name="ecart_au_plan"
        )

    # Angle 4 — Anticipation (sans copie)
    def _anticipation(self) -> pd.Series:
        upb_orig   = self.first[self.ORIG_UPB_COL]
        rate       = self.first[self.RATE_COL]
        n          = self.first[self.AGE_COL] + self.first[self.REMAIN_COL]
        r          = rate / 100 / 12
        rn         = (1 + r) ** n
        mensualite = upb_orig * r * rn / (rn - 1)
        mensualite = mensualite.where((r > 0) & (n > 0), np.nan)
        mensualite_map = mensualite.set_axis(self.first[self.LOAN_COL])

        self.df["_mensualite"] = self.df[self.LOAN_COL].map(mensualite_map)
        self.df["_remb"]       = self.df["_diff_upb"].abs()
        self.df["_sup"]        = self.df["_remb"] > self.df["_mensualite"]

        return (
            self.grp["_sup"]
            .mean()
            .rename("anticipation")
        )

    # Build
    def build(self) -> pd.DataFrame:
        niveau        =   self._niveau().reset_index(name="niveau")
        progression   =   self._progression().reset_index()
        ecart_au_plan =   self._ecart_au_plan().reset_index(name="ecart_au_plan")
        anticipation  =   self._anticipation().reset_index()

        return (
            niveau
            .merge(progression, on=self.LOAN_COL, how="left")
            .merge(ecart_au_plan, on=self.LOAN_COL, how="left")
            .merge(anticipation, on=self.LOAN_COL, how="left")
        )