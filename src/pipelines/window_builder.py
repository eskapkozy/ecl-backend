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

    LOAN_COL   = "LOAN_SEQUENCE_NUMBER"
    PERIOD_COL = "MONTHLY_REPORTING_PERIOD"

    def __init__(self, df: pd.DataFrame, window_months: int = 12):
        self.df            = df  # pas de copie — lecture seule
        self.window_months = window_months
        self._prepare()

    def _prepare(self):
        # Format explicite — plus rapide que format="mixed"
        if self.df[self.PERIOD_COL].dtype == object:
            self.df = self.df.copy()
            self.df[self.PERIOD_COL] = pd.to_datetime(
                self.df[self.PERIOD_COL], format="%Y-%m-%d", errors="coerce"
            )

        # Tri uniquement si nécessaire
        if not self.df[[self.LOAN_COL, self.PERIOD_COL]].equals(
            self.df[[self.LOAN_COL, self.PERIOD_COL]].sort_values(
                [self.LOAN_COL, self.PERIOD_COL]
            )
        ):
            self.df = self.df.sort_values([self.LOAN_COL, self.PERIOD_COL])

    def build(self) -> pd.DataFrame:
        return (
            self.df
            .groupby(self.LOAN_COL)
            .tail(self.window_months)
            .reset_index(drop=True)
        )