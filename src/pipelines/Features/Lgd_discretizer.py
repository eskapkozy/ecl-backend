import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np


# =============================================================================
# LGD QUANTILE DISCRETIZATION — Burakov (2026) Methodology
# =============================================================================
#
# RÉFÉRENCE : Burakov, D. (2026). Another Way to Model LGD — Probabilistic
#             Regression via Quantile-Based Classification.
#
# PRINCIPE :
#   Discrétiser la variable LGD continue en K bins quantiles, entraîner un
#   classifieur multiclasse sur ces K classes, puis reconstruire un point
#   estimate via :
#       ŷ = Σ p_k(X) · v_k    pour k = 1, ..., K
#   où v_k est le midpoint du bin k.
#
# POINT DE VIGILANCE CENTRAL (identifié avant implémentation) :
#   Le nombre de probabilités en sortie du classifieur est garanti égal à K
#   PAR CONSTRUCTION (num_class=K, softmax). Le vrai risque n'est pas le
#   nombre de probabilités, mais L'ALIGNEMENT entre :
#       - l'ordre des classes apprises par le modèle (0 à K-1)
#       - l'ordre des midpoints v_k utilisés au moment du produit scalaire
#
#   Cet objet (LGDDiscretizer) centralise bin edges ET midpoints ensemble,
#   pour qu'ils ne puissent jamais être désynchronisés entre train et
#   inférence. Persisté tel quel comme artifact MLflow.
#
# ANTI-LEAKAGE :
#   Le fit (calcul des bin edges) doit être appelé UNIQUEMENT sur le train.
#   transform() sur val/test/inférence réutilise les edges déjà appris,
#   ne refait jamais un fit.
#
# =============================================================================


class LGDDiscretizer:
    """
    Discrétise une variable LGD continue en K bins quantiles et fournit
    la transformation inverse (bin -> point estimate) de façon cohérente.

    Usage
    -----
    # Train — fit uniquement sur train
    discretizer = LGDDiscretizer(n_bins=8)
    y_train_bins = discretizer.fit_transform(y_train)

    # Val / Test / Inférence — transform uniquement, pas de fit
    y_test_bins = discretizer.transform(y_test)

    # Reconstruction du point estimate à partir des probabilités du modèle
    # proba.shape == (n_samples, n_bins) — sortie de model.predict_proba()
    lgd_hat = discretizer.expected_value(proba)
    """

    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins
        self.bin_edges_: np.ndarray | None = None
        self.midpoints_: np.ndarray | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit — UNIQUEMENT sur train
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series) -> "LGDDiscretizer":
        """
        Calcule les bin edges quantiles et les midpoints à partir de y.
        Ne doit être appelé que sur la target LGD du train set.
        """
        y = pd.Series(y).dropna()

        if y.empty:
            raise ValueError("LGDDiscretizer.fit() : série vide après dropna().")

        # qcut avec retbins pour récupérer les bornes exactes
        _, bin_edges = pd.qcut(
            y, q=self.n_bins, retbins=True, duplicates="drop"
        )

        effective_bins = len(bin_edges) - 1
        if effective_bins != self.n_bins:

            logger.info(msg=f"[LGDDiscretizer] n_bins ajusté : {self.n_bins} demandés → ")
            logger.info(msg=f"{effective_bins} bins effectifs (doublons dans la distribution).")

            print(
                f"[LGDDiscretizer] n_bins ajusté : {self.n_bins} demandés → ",
                f"{effective_bins} bins effectifs (doublons dans la distribution)."
            )
            self.n_bins = effective_bins

        # Forcer les bornes extrêmes à couvrir tout l'espace [0, 1] —
        # garantit qu'aucune valeur future (val/test/inférence) ne tombe
        # hors des bins appris, même si elle dépasse légèrement min/max du train.
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

        self.bin_edges_ = bin_edges
        self.midpoints_ = (bin_edges[:-1] + bin_edges[1:]) / 2
        self._is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Transform — réutilisable sur train, val, test, inférence
    # ------------------------------------------------------------------

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Assigne chaque valeur de y à un bin index (0 à n_bins-1) en
        réutilisant les bin edges déjà appris via fit().
        Ne refait jamais de fit — garantit l'absence de leakage.
        """
        self._check_is_fitted()

        bin_index = pd.cut(
            y,
            bins=self.bin_edges_,
            labels=False,
            include_lowest=True,
        )

        n_unassigned = bin_index.isna().sum()
        if n_unassigned > 0:
            raise ValueError(
                f"{n_unassigned} valeurs n'ont pu être assignées à aucun bin — "
                f"vérifier que les valeurs sont bien dans [0, 1]."
            )

        return pd.Series(bin_index.astype(int))

    def fit_transform(self, y: pd.Series) -> pd.Series:
        return self.fit(y).transform(y)

    # ------------------------------------------------------------------
    # Point estimate — Burakov eq. (3) : ŷ = Σ p_k(X) · v_k
    # ------------------------------------------------------------------

    def expected_value(self, proba: np.ndarray) -> np.ndarray:
        """
        Calcule le point estimate LGD à partir d'une matrice de
        probabilités multiclasse.

        Args:
            proba: array de shape (n_samples, n_bins), sortie de
                   model.predict_proba(). L'ordre des colonnes DOIT
                   correspondre à l'ordre des classes 0..n_bins-1
                   apprises par le modèle.

        Returns:
            array de shape (n_samples,) — point estimate ŷ par observation.

        Raises:
            ValueError si le nombre de colonnes de proba ne correspond pas
            au nombre de midpoints — garde explicite contre le
            désalignement bins/probabilités identifié comme risque central.
        """
        self._check_is_fitted()

        proba = np.asarray(proba)

        if proba.shape[1] != len(self.midpoints_):
            raise ValueError(
                f"Désalignement détecté : le modèle retourne {proba.shape[1]} "
                f"probabilités mais {len(self.midpoints_)} midpoints sont "
                f"persistés. Vérifier que le binning_process chargé "
                f"correspond bien au modèle chargé (même run MLflow)."
            )

        return proba @ self.midpoints_

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def bin_counts(self, y: pd.Series) -> pd.Series:
        """
        Retourne l'effectif par bin pour vérification de robustesse
        statistique avant entraînement (chaque bin doit avoir un
        effectif suffisant pour que le classifieur apprenne un signal).
        """
        bin_index = self.transform(y)
        return bin_index.value_counts().sort_index()

    def summary(self) -> pd.DataFrame:
        """Table de référence bin -> [edge_low, edge_high, midpoint]."""
        self._check_is_fitted()
        return pd.DataFrame({
            "bin": range(self.n_bins),
            "edge_low": self.bin_edges_[:-1],
            "edge_high": self.bin_edges_[1:],
            "midpoint": self.midpoints_,
        })

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "LGDDiscretizer n'est pas fit — appeler fit() sur le train "
                "avant transform() ou expected_value()."
            )


# =============================================================================
# Exemple d'usage attendu (à valider en review)
# =============================================================================
#
# from sklearn.model_selection import train_test_split
#
# y_train, y_test = train_test_split(lgd_df["lgd_target"], test_size=0.3)
#
# discretizer = LGDDiscretizer(n_bins=8)
# y_train_bins = discretizer.fit_transform(y_train)   # fit + transform train
# y_test_bins  = discretizer.transform(y_test)         # transform seul, test
#
# print(discretizer.summary())
# print(discretizer.bin_counts(y_train))               # vérifier robustesse
#
# # ... entraînement LightGBM multiclass sur (X_train, y_train_bins) ...
#
# proba = model.predict_proba(X_test)                  # shape (n, 8)
# lgd_hat = discretizer.expected_value(proba)           # ŷ = Σ p_k · v_k
#
# =============================================================================