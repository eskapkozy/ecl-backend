"""
Feature Engineering — Groupe Capital
======================================
Approche : Question-driven feature engineering
Responsabilité : calcul des features uniquement
                 (la fenêtre est gérée par WindowBuilder)

Colonnes requises dans le DataFrame entrant :
    - LOAN_SEQUENCE_NUMBER
    - CURRENT_ACTUAL_UPB
    - CURRENT_INTEREST_RATE
    - LOAN_AGE
    - REMAINING_MONTHS_TO_LEGAL_MATURITY
    - ORIGINAL_UPB  (depuis jointure origination — obligatoire)
"""

import pandas as pd
import numpy as np


class CapitalFeatures:
    """
    Calcule les features comportementales du groupe Capital.
    Attend un DataFrame déjà découpé en fenêtre par WindowBuilder.

    Usage
    -----
    from window_builder import WindowBuilder
    from capital_features import CapitalFeatures

    df_win   = WindowBuilder(df, window_months=12).build()
    features = CapitalFeatures(df_win).build()
    """

    LOAN_COL     = "LOAN_SEQUENCE_NUMBER"
    UPB_COL      = "CURRENT_ACTUAL_UPB"
    RATE_COL     = "CURRENT_INTEREST_RATE"
    AGE_COL      = "LOAN_AGE"
    REMAIN_COL   = "REMAINING_MONTHS_TO_LEGAL_MATURITY"
    ORIG_UPB_COL = "ORIGINAL_UPB"

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in [self.UPB_COL, self.RATE_COL, self.AGE_COL,
                    self.REMAIN_COL, self.ORIG_UPB_COL]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Angle 1 — Niveau
    # Question : quelle part du capital reste due ?
    # ------------------------------------------------------------------

    def _niveau(self, g: pd.DataFrame) -> float:
        """Ratio UPB courant / UPB origination."""
        upb_courant = g[self.UPB_COL].iloc[-1]
        upb_orig    = g[self.ORIG_UPB_COL].iloc[0]
        if upb_orig == 0 or np.isnan(upb_orig):
            return np.nan
        return upb_courant / upb_orig

    # ------------------------------------------------------------------
    # Angle 2 — Progression
    # Question : le capital baisse-t-il régulièrement ?
    # ------------------------------------------------------------------

    def _progression(self, g: pd.DataFrame) -> float:
        """Écart-type des variations mensuelles du UPB."""
        return g[self.UPB_COL].diff().dropna().std()

    # ------------------------------------------------------------------
    # Angle 3 — Écart au plan
    # Question : est-on en retard sur le remboursement théorique ?
    # ------------------------------------------------------------------

    def _ecart_au_plan(self, g: pd.DataFrame) -> float:
        """
        UPB théorique - UPB réel au point d'observation.
        Valeur positive = client en retard sur son plan.
        """
        row         = g.iloc[-1]
        upb_orig    = row[self.ORIG_UPB_COL]
        age         = row[self.AGE_COL]
        remaining   = row[self.REMAIN_COL]
        rate_annual = row[self.RATE_COL]

        if any(pd.isna([upb_orig, age, remaining, rate_annual])):
            return np.nan

        n = age + remaining
        r = rate_annual / 100 / 12

        if r == 0:
            upb_theorique = upb_orig * (1 - age / n) if n > 0 else np.nan
        else:
            upb_theorique = upb_orig * (
                ((1 + r) ** n - (1 + r) ** age) / ((1 + r) ** n - 1)
            )

        return upb_theorique - row[self.UPB_COL]

    # ------------------------------------------------------------------
    # Angle 4 — Anticipation
    # Question : rembourse-t-il plus que prévu ?
    # ------------------------------------------------------------------

    def _anticipation(self, g: pd.DataFrame) -> float:
        """Ratio de mois où le remboursement dépasse la mensualité théorique."""
        upb_orig    = g[self.ORIG_UPB_COL].iloc[0]
        rate_annual = g[self.RATE_COL].iloc[0]
        age_0       = g[self.AGE_COL].iloc[0]
        remaining_0 = g[self.REMAIN_COL].iloc[0]

        if any(pd.isna([upb_orig, rate_annual, age_0, remaining_0])):
            return np.nan

        n = age_0 + remaining_0
        r = rate_annual / 100 / 12

        if r == 0 or n == 0:
            return np.nan

        mensualite       = upb_orig * r * (1 + r) ** n / ((1 + r) ** n - 1)
        remboursements   = g[self.UPB_COL].diff().abs().dropna()
        nb_superieur     = (remboursements > mensualite).sum()

        return nb_superieur / len(remboursements) if len(remboursements) > 0 else np.nan

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Retourne une table agrégée — une ligne par prêt —
        avec toutes les features du groupe Capital.
        """
        return (
            self.df.groupby(self.LOAN_COL)
            .apply(lambda g: pd.Series({
                "niveau"        : self._niveau(g),
                "progression"   : self._progression(g),
                "ecart_au_plan" : self._ecart_au_plan(g),
                "anticipation"  : self._anticipation(g),
            }))
            .reset_index()
        )