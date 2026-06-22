"""
EDA Utility — Freddie Mac SFLLD
=================================
Classe utilitaire pour l'analyse exploratoire des données.
La target n'est pas requise à l'instanciation — elle est passée
en paramètre dans les méthodes bivariées.

Usage
-----
util = Util(aggregated_features)

# Univarié
util.global_view("freq")
util.visualize_cat("credit_segment")

# Bivarié (une fois la target construite)
util.box_barPlot(target="default", feature="freq")
util.scatter_of_default_qtl(feature="freq", target="default")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class Util:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy()

    def bivariate_summary(self, target: str) -> pd.DataFrame:
        """
        Teste la significativité de chaque feature numérique vis-à-vis de la target.
        Retourne une matrice triée par p-value croissante.
        """
        from scipy.stats import mannwhitneyu

        numeric_cols = self.dataset.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target]

        results = []
        for col in numeric_cols:
            g0 = self.dataset[self.dataset[target] == 0][col].dropna()
            g1 = self.dataset[self.dataset[target] == 1][col].dropna()

            stat, p = mannwhitneyu(g0, g1, alternative="two-sided")

            results.append({
                "feature": col,
                "mean_0": g0.mean(),
                "mean_1": g1.mean(),
                "median_0": g0.median(),
                "median_1": g1.median(),
                "stat": stat,
                "p_value": p,
                "significatif": p < 0.05
            })

        return pd.DataFrame(results).sort_values("p_value")


    # ------------------------------------------------------------------
    # Univarié — numérique
    # ------------------------------------------------------------------

    def global_view(self, feature: str):
        """Distribution + boxplot d'une feature numérique."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(data=self.dataset, x=feature, kde=True, bins=20, ax=axes[0])
        axes[0].set_title(f'skew: {self.dataset[feature].skew():.2f}')

        sns.boxplot(self.dataset, x=feature, ax=axes[1])
        axes[1].set_title(f'median : {self.dataset[feature].median():.2f}')

        plt.tight_layout()
        plt.show()

    def binsFrom_hist(self, feature: str, bins: int = 20) -> pd.DataFrame:
        """Retourne un DataFrame des intervalles de fréquence."""
        counts, bin_edges = np.histogram(self.dataset[feature].dropna(), bins=bins)

        return pd.DataFrame({
            'intervalle'  : [f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]"
                             for i in range(len(counts))],
            'borne_gauche': bin_edges[:-1],
            'borne_droite': bin_edges[1:],
            'frequence'   : counts,
            'pct'         : (counts / counts.sum() * 100).round(2)
        }).sort_values('frequence', ascending=False)

    def probdensity(self, feature: str, law: str, upper_born, lower_born) -> float:
        """Probabilité d'appartenance à un intervalle selon une loi."""
        if law == 'log-normal':
            shape, loc, scale = stats.lognorm.fit(self.dataset[feature].dropna())
            dist = stats.lognorm(shape, loc, scale)
        else:
            mu, sigma = stats.norm.fit(self.dataset[feature].dropna())
            dist = stats.norm(mu, sigma)
        return dist.cdf(upper_born) - dist.cdf(lower_born)

    # ------------------------------------------------------------------
    # Univarié — catégoriel
    # ------------------------------------------------------------------

    def visualize_cat(self, feature: str, figsize: tuple = (10, 5)):
        """Countplot + pie chart d'une feature catégorielle."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        count = self.dataset[feature].value_counts()

        sns.countplot(self.dataset, x=feature, ax=axes[0])
        axes[1].pie(count, labels=count.index, radius=1.2, autopct='%1.1f%%')

        plt.tight_layout()
        plt.show()

    def visualize_dates(self, date_col: str, figsize: tuple = (12, 4)):
        """
        Visualise la distribution temporelle d'une colonne de dates.
        Deux vues : évolution mensuelle + distribution annuelle.
        """
        dates = pd.to_datetime(self.dataset[date_col], errors="coerce")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Vue mensuelle
        dates.value_counts().sort_index().plot(
            kind="line", ax=axes[0], title=f"{date_col} — distribution mensuelle"
        )

        # Vue annuelle
        dates.dt.year.value_counts().sort_index().plot(
            kind="bar", ax=axes[1], title=f"{date_col} — distribution annuelle"
        )

        plt.tight_layout()
        plt.show()


    # ------------------------------------------------------------------
    # Discrétisation
    # ------------------------------------------------------------------

    def qcut(self, feature: str, q: int = 4, labels: list = None):
        """Découpage en quantiles — retourne (DataFrame enrichi, bornes)."""
        if labels is None:
            labels = ['FAIBLE', 'MOYEN', 'HAUT', 'FORT']
        concat_ = feature + '_TRANCHE'
        x_df = self.dataset.copy()
        x_df[concat_], limit = pd.qcut(
            x=x_df[feature], q=q, labels=labels, retbins=True, duplicates='drop'
        )
        return x_df, limit

    def qdiscret_proportion(self, feature: str, other_feature: str,
                             q: int = 4, labels: list = None):
        """Proportion de chaque modalité de other_feature par tranche de feature."""
        df, limit = self.qcut(feature, q, labels)
        grouped = df.groupby([feature + '_TRANCHE', other_feature]).size()
        proportion = (
            grouped
            .groupby(level=0)
            .apply(lambda x: x / x.sum())
            .rename('proportion')
        )
        return proportion, limit

    # ------------------------------------------------------------------
    # Bivarié — target en paramètre
    # ------------------------------------------------------------------

    def box_barPlot(self, target: str, feature: str):
        """
        Boxplot + barplot d'une feature numérique selon la target.
        La target doit être binaire (0/1).
        """
        y_mean   = self.dataset.groupby(target)[feature].mean()
        y_median = self.dataset.groupby(target)[feature].median()

        cats = sorted(self.dataset[target].dropna().unique())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(data=self.dataset, x=target, y=feature, hue=target, ax=axes[0])
        axes[0].set_title(
            f'median_1: {y_median.get(1, y_median.iloc[-1]):.2f}   '
            f'median_0: {y_median.get(0, y_median.iloc[0]):.2f}'
        )

        sns.barplot(data=self.dataset, x=target, y=feature, hue=target, ax=axes[1])
        axes[1].set_title(
            f'mean_1: {y_mean.get(1, y_mean.iloc[-1]):.2f}   '
            f'mean_0: {y_mean.get(0, y_mean.iloc[0]):.2f}'
        )

        plt.tight_layout()
        plt.show()

    def scatter_of_default_qtl(self, feature: str, target: str,
                                other_feature: str = None,
                                index: list = None):
        """
        Visualisation de la distribution par quartile selon la target.
        """
        if other_feature is None:
            other_feature = target
        if index is None:
            index = ['FAIBLE', 'MOYEN', 'HAUT', 'FORT']

        cats = sorted(self.dataset[target].dropna().unique())

        prop, _ = self.qdiscret_proportion(feature, other_feature)

        disc_1 = prop[:, :, cats[-1]].values
        disc_0 = prop[:, :, cats[0]].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.stripplot(x=index, y=disc_1, linestyles="-", ax=axes[0])
        axes[0].set_title(f'{target} = {cats[-1]}')

        sns.stripplot(x=index, y=disc_0, linestyles='-', ax=axes[1])
        axes[1].set_title(f'{target} = {cats[0]}')

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Tableau de contingence
    # ------------------------------------------------------------------

    def contingency_count(self, line_index: str, feature: str) -> pd.DataFrame:
        return pd.crosstab(self.dataset[line_index], self.dataset[feature], margins=True)

    def contingency_effLine(self, line_index: str, feature: str) -> pd.DataFrame:
        return pd.crosstab(self.dataset[line_index], self.dataset[feature],
                           margins=True, normalize='all')

    def contingency_perLine(self, line_index: str, feature: str) -> pd.DataFrame:
        return pd.crosstab(self.dataset[line_index], self.dataset[feature],
                           normalize='columns', margins=True)