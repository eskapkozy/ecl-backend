"""
LGD Run — Contrat intermédiaire
=================================
Parent intermédiaire entre RunAbstraction et les implémentations concrètes
(LightGBMLGDRun, ...) pour le modèle LGD — Probabilistic Regression via
Quantile-Based Classification (Burakov, 2026).

Pourquoi un parent distinct de PDRun :
    PDRun encode un contrat de classification binaire (threshold search,
    recall/precision/F1, SMOTE, calibration sigmoid sur probabilité de
    défaut). Aucun de ces éléments n'a de sens pour LGD :
        - pas de classe positive/négative — 8 classes ordinales (bins)
        - pas de déséquilibre binaire à corriger par SMOTE
        - pas de seuil de décision — on reconstruit un point estimate continu

Ce que LGDRun ajoute au contrat RunAbstraction :
    - gestion du LGDDiscretizer (fit sur train, transform sur val/test/infer)
      — symétrique au binning_process WoE dans PDRun
    - calcul du point estimate ŷ = Σ p_k(X) · v_k
    - double famille de métriques :
        * classification multiclasse (diagnostic du classifieur) :
          log-loss multiclasse, accuracy, confusion matrix 8x8
        * régression / calibration sur le point estimate (ce qui compte
          réellement pour juger le modèle LGD, cf. Burakov Table 3) :
          RMSE, Dxy (Somers' D), ECE par bin

Hiérarchie
----------
RunAbstraction
        ↓
LGDRun (ce fichier)            → contrat discretizer, métriques hybrides
        ↓
LightGBMLGDRun, ...             → implémentation concrète de l'algorithme
"""
from abc import abstractmethod
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, mean_squared_error


from src.Utile.artifactManager import ArtifactManager, ArtifactType
from src.runAbstraction import RunAbstraction
from src.pipelines.Features.Lgd_discretizer import LGDDiscretizer


class LGDRun(RunAbstraction):

    def __init__(self, train_map: dict = None, test_map: dict = None,
                 val_map: dict = None, config_path: str = None, test_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path, test_path)

        self.discretizer: LGDDiscretizer | None = None
        self.featurePipeline = None

        # Bin index (target d'entraînement multiclasse), distinct de la
        # target LGD continue originale (_y_train, _y_val, _y_test).
        self.y_train_bins = None
        self.y_val_bins   = None
        self.y_test_bins  = None


        self._artifact_manager = ArtifactManager()

        self._rmse = None
        self._dxy  = None
        self._ece  = None

    @abstractmethod
    def setup(self, featurePipeline):
        raise NotImplementedError


    # ------------------------------------------------------------------
    # Discretizer — symétrique au binning_process WoE dans PDRun
    # ------------------------------------------------------------------

    def _fit_discretizer(self, y_train: pd.Series, n_bins: int = 8) -> LGDDiscretizer:
        """
        Fit UNIQUEMENT sur train. À appeler dans _run_train() des sous-classes.
        """
        discretizer = LGDDiscretizer(n_bins=n_bins)
        self.y_train_bins = discretizer.fit_transform(y_train)
        self.discretizer = discretizer
        return discretizer

    def _transform_with_discretizer(self, y: pd.Series, discretizer: LGDDiscretizer = None) -> pd.Series:
        """
        Transform sur val/test/inférence — réutilise le discretizer déjà fit.
        Ne refait jamais de fit (anti-leakage, symétrique à apply_woe côté PD).
        """
        d = discretizer or self.discretizer
        if d is None:
            raise RuntimeError(
                "discretizer manquant — appeler _fit_discretizer(y_train) d'abord, "
                "ou charger un discretizer persisté via artifact."
            )
        return d.transform(y)

    # ------------------------------------------------------------------
    # Point estimate — Burakov eq. (3)
    # ------------------------------------------------------------------

    def _compute_point_estimate(self, proba: np.ndarray, discretizer: LGDDiscretizer = None) -> np.ndarray:
        """
        ŷ = Σ p_k(X) · v_k

        La garde d'alignement bins/probabilités est déléguée à
        LGDDiscretizer.expected_value() — voir lgd_discretizer.py.
        """
        d = discretizer or self.discretizer
        if d is None:
            raise RuntimeError("discretizer manquant pour le calcul du point estimate.")
        return d.expected_value(proba)

    # ------------------------------------------------------------------
    # Métriques — Famille 1 : classification multiclasse (diagnostic)
    # ------------------------------------------------------------------

    def _classification_metrics(self, y_true_bins: np.ndarray, proba: np.ndarray) -> dict:
        """
        Diagnostic du classifieur multiclasse en tant que tel — utile pour
        détecter un classifieur dégénéré (ex. qui prédit toujours le même
        bin), indépendamment de la qualité du point estimate reconstruit.
        """
        y_pred_bins = np.argmax(proba, axis=1)

        metrics = {
            "multiclass_log_loss": log_loss(y_true_bins, proba, labels=list(range(proba.shape[1]))),
            "multiclass_accuracy": accuracy_score(y_true_bins, y_pred_bins),
        }

        cm = confusion_matrix(y_true_bins, y_pred_bins, labels=list(range(proba.shape[1])))
        metrics["confusion_matrix"] = cm

        return metrics

    # ------------------------------------------------------------------
    # Métriques — Famille 2 : régression / calibration sur le point estimate
    # ------------------------------------------------------------------

    @staticmethod
    def _somers_d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Somers' D (Dxy) — mesure de rank-ordering standard pour les modèles
        LGD (cf. Burakov, Table 3). Calculé à partir de la corrélation de
        rang de Kendall (tau-a), relation directe : Dxy = tau-a sous
        l'absence d'ex-aequo significatifs.

        Implémentation via scipy pour éviter une dépendance supplémentaire.
        """
        from scipy.stats import kendalltau
        tau, _ = kendalltau(y_true, y_pred)
        return tau

    @staticmethod
    def _expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                                      n_bins: int = 5) -> float:
        """
        ECE sur le point estimate — compare la proportion observée vs
        prédite par bin de LGD (réplique la logique des Figures 2/3 de
        Burakov, indépendamment du binning d'entraînement K=8 : ici on
        rebinne sur des intervalles égaux [0,1] pour la lecture de
        calibration, pas pour l'entraînement).
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(y_pred, bin_edges) - 1, 0, n_bins - 1)

        ece = 0.0
        n_total = len(y_true)

        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            observed = y_true[mask].mean()
            predicted = y_pred[mask].mean()
            weight = mask.sum() / n_total
            ece += weight * abs(observed - predicted)

        return ece

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Métriques sur le point estimate ŷ reconstruit — ce qui compte
        réellement pour juger le modèle LGD, symétrique à la Table 3
        de Burakov (Dxy, RMSE) + ECE (Figures 2/3).
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        dxy = self._somers_d(y_true, y_pred)
        ece = self._expected_calibration_error(y_true, y_pred)

        return {
            "rmse": rmse,
            "dxy": dxy,
            "ece": ece,
        }

    def _evaluate(self, y_true_continuous: np.ndarray, y_true_bins: np.ndarray,
                      proba: np.ndarray, discretizer: LGDDiscretizer = None) -> dict:
        """
        Orchestration — calcule le point estimate puis les deux familles
        de métriques. À appeler depuis _run_train()/_run_test() des
        sous-classes concrètes, symétrique à la structure observée dans
        XGBoostRun (PD) pour la cohérence de logging MLflow.
        """
        y_pred = self._compute_point_estimate(proba, discretizer)

        metrics = {}
        metrics.update(self._classification_metrics(y_true_bins, proba))
        metrics.update(self._regression_metrics(y_true_continuous, y_pred))
        metrics["y_pred"] = y_pred

        self._rmse = metrics["rmse"]
        self._dxy = metrics["dxy"]
        self._ece = metrics["ece"]

        #self._multiclass_accuracy = metrics["multiclass_accuracy"]
        #self._multiclass_log_loss = metrics["multiclass_log_loss"]

        return metrics

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError


    def _load_data(self):
        """Charge ou prépare les données nécessaires au run (hors maps déjà fournies)."""
        pass

    # ------------------------------------------------------------------
    # Artefact — discretizer (edges + midpoints), symétrique au
    # binning_process logué côté PD
    # ------------------------------------------------------------------




    def _log_feature_artifact(self):
        self._artifact_manager.log(
            obj=self.discretizer,
            name="lgd_discretizer",
            artifact_type=ArtifactType.PKL
        )
        self._artifact_manager.log(
            obj=self.discretizer.summary(),
            name="lgd_bin_summary",
            artifact_type=ArtifactType.CSV
        )

    def _log_calibration_by_bin(self, y_true_bins: np.ndarray, proba: np.ndarray):
        n_bins = proba.shape[1]
        observed = np.array([(y_true_bins == k).mean() for k in range(n_bins)])
        predicted = proba.mean(axis=0)

        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(n_bins)
        width = 0.35

        ax.bar(x - width / 2, observed, width, label="Observed", color="#1f9e89")
        ax.bar(x + width / 2, predicted, width, label="Predicted", color="#cc4c5f")

        ece = self._ece if self._ece is not None else float("nan")
        ax.set_title(f"Realized vs Estimated LGD per bin (ECE={ece:.4f})")
        ax.set_xlabel("LGD bin")
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_path = "lgd_calibration_by_bin.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fig_path)

    def save_evaluation_metrics(self, test_config_path: str):

        path = Path(test_config_path)

        with open(path, "r") as f:
            test_config = yaml.safe_load(f)

        test_config["evaluation"] = {
            "regression_result": {
                "rmse": {
                    "value": float(self._rmse)
                },
                "dxy": {
                    "value": float(self._dxy)
                }
            },
            "calibration_result": {
                "ece": {
                    "value": float(self._ece)
                }
            },

        }

        with open(path, "w") as f:
            yaml.safe_dump(
                test_config,
                f,
                sort_keys=False,
                default_flow_style=False,
                indent=2
            )

    '''
    
        @abstractmethod
    def _discretizer_artifact_type(self):
        """
        Type d'artifact MLflow pour persister le discretizer (probablement
        ArtifactType.PKL — à confirmer selon ArtifactManager). Laissé
        abstrait pour cohérence avec le pattern _build_custom_scorer /
        _run_grid_search déjà abstrait dans RunAbstraction.
        """
        raise NotImplementedError
    
    
    '''









    # ------------------------------------------------------------------
    # Contrat hérité de RunAbstraction — non implémenté ici,
    # délégué aux sous-classes concrètes (LightGBMLGDRun, ...)
    # ------------------------------------------------------------------
    #   setup(), _load_data(), _load_scaler(), _run_train(), _run_test(),
    #   _build_custom_scorer(), _run_grid_search(), save_evaluation_metrics()