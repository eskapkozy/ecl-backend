"""
Feature Engineering — Groupe Retard (vectorisé v3)
====================================================
Optimisation v3 :
    - Un seul groupby.agg pour freq, severite, profondeur_max, n_profondeur_max
    - Transforms partagés calculés une seule fois dans __init__
    - Tendance et récupération sur self.df déjà enrichi
"""

import pandas as pd
import numpy as np
import time

class DelinquencyFeatures:

    LOAN_COL = "LOAN_SEQUENCE_NUMBER"
    DPD_COL  = "CURRENT_LOAN_DELINQUENCY_STATUS"

    def __init__(self, df: pd.DataFrame):

        t0 = time.time()
        self.df = df.copy()
        print(f"Copy DataFrame     : {time.time()-t0:.1f}s")


        t0 = time.time()
        # Cast DPD
        self.df[self.DPD_COL] = (
            pd.to_numeric(self.df[self.DPD_COL], errors="coerce").fillna(0)
        )
        print(f"Cast DPD           : {time.time()-t0:.1f}s")


        t0 = time.time()
        self.gpr = self.df.groupby(self.LOAN_COL)
        print(f"Groupby            : {time.time()-t0:.1f}s")


        t0 = time.time()
        # Colonnes de travail — calculées une seule fois
        self.df["_t"]         = self.gpr.cumcount()
        self.df["_en_retard"] = (self.df[self.DPD_COL] > 0).astype(int)
        self.df["_max_dpd"]   = self.gpr[self.DPD_COL].transform("max")
        self.df["_t_mean"]    = self.gpr["_t"].transform("mean")
        self.df["_y_mean"]    = self.gpr[self.DPD_COL].transform("mean")
        print(f"Colonnes de travail: {time.time()-t0:.1f}s")


        # Shift pour récupération — calculé une seule fois
        self.df["_shift"]  = self.gpr["_en_retard"].shift(1).fillna(0)
        self.df["_debut"]  = ((self.df["_en_retard"] == 1) & (self.df["_shift"] == 0)).astype(int)

    # ------------------------------------------------------------------
    # Agrégation centralisée — un seul groupby
    # ------------------------------------------------------------------



    def _agg_principal(self) -> pd.DataFrame:
        """
        Calcule freq, severite, profondeur_max, n_profondeur_max
        en un seul groupby.agg.
        """
        self.df["_at_max"] = (
            (self.df[self.DPD_COL] == self.df["_max_dpd"]) &
            (self.df["_max_dpd"] > 0)
        ).astype(int)

        return self.gpr.agg(
            freq            = ("_en_retard", "mean"),
            severite        = (self.DPD_COL, "mean"),
            profondeur_max  = (self.DPD_COL, "max"),
            n_profondeur_max= ("_at_max", "sum"),
        ).reset_index()

    # ------------------------------------------------------------------
    # Tendance — slope vectorisé
    # ------------------------------------------------------------------

    def _tendance(self) -> pd.Series:
        num = (
            ((self.df["_t"] - self.df["_t_mean"]) *
             (self.df[self.DPD_COL] - self.df["_y_mean"]))
            .groupby(self.df[self.LOAN_COL]).sum()
        )
        den = (
            ((self.df["_t"] - self.df["_t_mean"]) ** 2)
            .groupby(self.df[self.LOAN_COL]).sum()
        )
        return (num / den.replace(0, np.nan)).rename("tendance")

    # ------------------------------------------------------------------
    # Récupération — durée moyenne des épisodes de retard
    # ------------------------------------------------------------------

    def _recuperation(self) -> pd.Series:
        episode = (
            self.gpr["_debut"]
            .cumsum()
            .where(self.df["_en_retard"] == 1)
        )

        dur = (
            episode
            .groupby([self.df[self.LOAN_COL], episode])
            .transform("count")
            .where(self.df["_en_retard"] == 1)
        )

        return (
            dur.groupby(self.df[self.LOAN_COL])
            .mean()
            .rename("recuperation")
        )

    # ------------------------------------------------------------------
    # Combinaisons
    # ------------------------------------------------------------------

    def _combinaisons(self, agg: pd.DataFrame) -> pd.DataFrame:
        agg["freq_x_profondeur_max"] = agg["freq"] * agg["profondeur_max"]
        agg["freq_x_tendance"]       = agg["freq"] * agg["tendance"]
        agg["freq_x_recuperation"]   = agg["freq"] * agg["recuperation"]
        agg["recidivisme_extreme"]   = agg["n_profondeur_max"] * agg["profondeur_max"]
        return agg

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Retourne une table agrégée — une ligne par prêt.
        Un seul groupby.agg pour les métriques principales.
        """
        agg = self._agg_principal()

        tendance     = self._tendance().reset_index()
        recuperation = self._recuperation().reset_index()

        agg = (
            agg
            .merge(tendance,     on=self.LOAN_COL, how="left")
            .merge(recuperation, on=self.LOAN_COL, how="left")
        )

        return self._combinaisons(agg)