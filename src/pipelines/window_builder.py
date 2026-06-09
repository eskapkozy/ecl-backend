"""
Window Builder
==============
Responsabilité unique : découper le panel mensuel en fenêtres d'observation.

Ce module ne calcule aucune feature.
Il prépare le DataFrame pour être consommé par les modules de features.

Usage
-----
from window_builder import WindowBuilder

wb     = WindowBuilder(df, window_months=12)
df_win = wb.build()
"""

import pandas as pd


class WindowBuilder:
    """
    Découpe un panel mensuel de prêts en fenêtres glissantes.

    Paramètres
    ----------
    df             : DataFrame Freddie Mac historique — déjà chargé
    window_months  : longueur de la fenêtre d'observation (défaut 12)
    """

    LOAN_COL   = "LOAN_SEQUENCE_NUMBER"
    PERIOD_COL = "MONTHLY_REPORTING_PERIOD"

    def __init__(self, df: pd.DataFrame, window_months: int = 12):
        self.df            = df.copy()
        self.window_months = window_months
        self._prepare()

    # ------------------------------------------------------------------
    # Préparation
    # ------------------------------------------------------------------

    def _prepare(self):
        """Parse les dates et trie le panel chronologiquement."""
        self.df[self.PERIOD_COL] = pd.to_datetime(
            self.df[self.PERIOD_COL], format="%m%Y"
        )
        self.df = self.df.sort_values([self.LOAN_COL, self.PERIOD_COL])

        # Conversion DPD Freddie Mac (mois → jours) faite ici une seule fois
        if "CURRENT_LOAN_DELINQUENCY_STATUS" in self.df.columns:
            self.df["DPD_DAYS"] = (
                pd.to_numeric(
                    self.df["CURRENT_LOAN_DELINQUENCY_STATUS"], errors="coerce"
                )
                .fillna(0)
                .clip(lower=0)
                * 30
            )

    # ------------------------------------------------------------------
    # Fenêtre glissante
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Retourne le panel réduit aux N derniers mois par prêt.
        Chaque prêt conserve au maximum window_months lignes.
        """
        return (
            self.df
            .groupby(self.LOAN_COL, group_keys=False)
            .apply(lambda g: g.tail(self.window_months))
            .reset_index(drop=True)
        )