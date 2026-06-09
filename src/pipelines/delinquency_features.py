"""
Feature Engineering — Groupe Retard
=====================================
Approche : Question-driven feature engineering
Responsabilité : calcul des features uniquement
                 (la fenêtre est gérée par WindowBuilder)

Colonnes requises dans le DataFrame entrant :
    - LOAN_SEQUENCE_NUMBER
    - DPD_DAYS  (produit par WindowBuilder)
"""

import pandas as pd
import numpy as np
from scipy import stats


class DelinquencyFeatures:
    """
    Calcule les features comportementales du groupe Retard.
    Attend un DataFrame déjà découpé en fenêtre par WindowBuilder.

    Usage
    -----
    from window_builder import WindowBuilder
    from delinquency_features import DelinquencyFeatures

    df_win   = WindowBuilder(df, window_months=12).build()
    features = DelinquencyFeatures(df_win).build()
    """

    LOAN_COL = "LOAN_SEQUENCE_NUMBER"
    DPD_COL  = "DPD_DAYS"

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------
    # Angle 1 — Fréquence
    # Question : combien de fois le client est-il en retard ?
    # ------------------------------------------------------------------

    def _frequence(self, g: pd.DataFrame) -> float:
        """Ratio mois en retard / total mois observés."""
        return (g[self.DPD_COL] > 0).sum() / len(g)

    # ------------------------------------------------------------------
    # Angle 2 — Sévérité
    # Question : en moyenne jusqu'où vont les retards ?
    # ------------------------------------------------------------------

    def _severite(self, g: pd.DataFrame) -> float:
        """DPD moyen sur la fenêtre."""
        return g[self.DPD_COL].mean()

    # ------------------------------------------------------------------
    # Angle 3 — Tendance
    # Question : les retards augmentent-ils ou diminuent-ils ?
    # ------------------------------------------------------------------

    def _tendance(self, g: pd.DataFrame) -> float:
        """
        Slope d'une régression linéaire sur la séquence DPD.
        Positif = dégradation / Négatif = amélioration / ~0 = stable.
        """
        x = np.arange(len(g))
        y = g[self.DPD_COL].values
        if y.std() == 0:
            return 0.0
        slope, *_ = stats.linregress(x, y)
        return slope

    # ------------------------------------------------------------------
    # Angle 4 — Récupération
    # Question : le client parvient-il à revenir à 0 DPD ?
    # ------------------------------------------------------------------

    def _recuperation(self, g: pd.DataFrame) -> float:
        """
        Nombre moyen de mois pour revenir à 0 DPD après un épisode de retard.
        Retourne NaN si aucun retard observé.
        """
        dpd    = g[self.DPD_COL].values
        delais = []
        i      = 0
        while i < len(dpd):
            if dpd[i] > 0:
                j = i + 1
                while j < len(dpd) and dpd[j] > 0:
                    j += 1
                if j < len(dpd):
                    delais.append(j - i)
                i = j
            else:
                i += 1
        return float(np.mean(delais)) if delais else np.nan

    # ------------------------------------------------------------------
    # Angle 5 — Profondeur maximale
    # Question : quel est le pire DPD jamais atteint ?
    # ------------------------------------------------------------------

    def _profondeur_max(self, g: pd.DataFrame) -> float:
        """DPD maximum atteint sur la fenêtre."""
        return g[self.DPD_COL].max()

    def _n_profondeur_max(self, g: pd.DataFrame) -> int:
        """Nombre de mois où le DPD maximum a été atteint (récidivisme)."""
        max_dpd = g[self.DPD_COL].max()
        if max_dpd == 0:
            return 0
        return int((g[self.DPD_COL] == max_dpd).sum())

    # ------------------------------------------------------------------
    # Combinaisons
    # ------------------------------------------------------------------

    def _combinaisons(self, row: pd.Series) -> pd.Series:
        """Features combinées — amplification mutuelle des signaux."""
        freq     = row["freq"]
        prof_max = row["profondeur_max"]
        n_max    = row["n_profondeur_max"]
        tendance = row["tendance"]
        recup    = row["recuperation"]

        return pd.Series({
            "freq_x_profondeur_max" : freq * prof_max,
            "freq_x_tendance"       : freq * tendance,
            "freq_x_recuperation"   : freq * recup if not np.isnan(recup) else np.nan,
            "recidivisme_extreme"   : n_max * prof_max,
        })

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Retourne une table agrégée — une ligne par prêt —
        avec toutes les features du groupe Retard.
        """
        agg = (
            self.df.groupby(self.LOAN_COL)
            .apply(lambda g: pd.Series({
                "freq"             : self._frequence(g),
                "severite"         : self._severite(g),
                "tendance"         : self._tendance(g),
                "recuperation"     : self._recuperation(g),
                "profondeur_max"   : self._profondeur_max(g),
                "n_profondeur_max" : self._n_profondeur_max(g),
            }))
            .reset_index()
        )

        combinaisons = agg.apply(self._combinaisons, axis=1)
        return pd.concat([agg, combinaisons], axis=1)