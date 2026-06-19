"""
PD Run
=======
Contrat spécifique au modèle PD — hérite de RunAbstraction.

Ajoute :
    - logging des artefacts WoE (binning_process, IV report, table, sélection)
    - figures d'évaluation classification (ROC, Precision-Recall)
    - sauvegarde des métriques PD (seuil, recall, precision, f1) dans la config test

Reste abstrait sur _load_data, _run_train, _run_test —
laissés aux implémentations concrètes (LogisticRegressionRun, XGBoostRun, ...).

Hiérarchie
----------
RunAbstraction
        ↓
PDRun (ce fichier)
        ↓
LogisticRegressionRun(PDRun), XGBoostRun(PDRun), ...
"""

from abc import abstractmethod
from pathlib import Path

import mlflow
import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, recall_score, precision_score, f1_score, \
    make_scorer

from src.runAbstraction import RunAbstraction
from src.Utile.artifactManager import ArtifactType
from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline


class PDRun(RunAbstraction):

    def __init__(self, train_map: dict = None, test_map: dict = None,
                 val_map: dict = None, config_path: str = None,test_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path,test_path)

        self.featurePipeline : PDFeaturePipeline = None

        # Métriques de classification — choisies lors de l'optimisation du seuil
        self._chosen_threshold = None
        self._best_recall      = None
        self._best_precision   = None
        self._best_f1          = None

    # ------------------------------------------------------------------
    # Contrat — toujours à implémenter par l'algorithme concret
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _run_train(self):
        raise NotImplementedError

    @abstractmethod
    def _run_test(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Artefacts — spécifiques au scoring PD / WoE
    # ------------------------------------------------------------------

    def _log_feature_artifact(self, feature_pipeline: PDFeaturePipeline):
        """Log le binning_process et les rapports WoE issus du pipeline PD."""
        self._artifact_manager.log(
            obj=feature_pipeline.binning_process,
            name="binning_process",
            artifact_type=ArtifactType.PKL
        )

        woe_pipeline = feature_pipeline.woe_pipeline_

        self._artifact_manager.log(
            obj=woe_pipeline.iv_report(),
            name="woe_iv_report",
            artifact_type=ArtifactType.JSON
        )

        self._artifact_manager.log(
            obj=woe_pipeline.selection_report(),
            name="corr_and_woe_selection_report",
            artifact_type=ArtifactType.JSON
        )

    # ------------------------------------------------------------------
    # Figures — évaluation classification binaire
    # ------------------------------------------------------------------

    def _log_roc_curve(self, y_data, y_proba):
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_data, y_proba, ax=ax, name="PD Model")
        ax.set_title("ROC Curve")
        mlflow.log_figure(fig, "plots/roc_curve.png")
        plt.close(fig)

    def _log_precision_recall_curve(self, y_data, y_proba):
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_data, y_proba, ax=ax, name="PD Model")
        ax.set_title("Precision-Recall Curve")
        mlflow.log_figure(fig, "plots/precision_recall_curve.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Métriques PD — seuil optimisé sous contrainte recall
    # ------------------------------------------------------------------

    def _set_pd_evaluation_metrics(self, chosen_threshold, best_recall,
                                   best_precision, best_f1):
        self._chosen_threshold = chosen_threshold
        self._best_recall      = best_recall
        self._best_precision   = best_precision
        self._best_f1          = best_f1

    def save_evaluation_metrics(self, test_config_path: str):

        path = Path(test_config_path)

        with open(path, "r") as f:
            test_config = yaml.safe_load(f)

        test_config["evaluation"] = {
            "threshold_result": {
                "threshold": float(self._chosen_threshold)
            },
            "constraints_result": {
                "recall": {
                    "value": float(self._best_recall)
                },
                "f1": {
                    "value": float(self._best_f1)
                }
            }
        }

        with open(path, "w") as f:
            yaml.safe_dump(
                test_config,
                f,
                sort_keys=False,
                default_flow_style=False,  # IMPORTANT: format bloc YAML lisible
                indent=2  # indentation propre et stable
            )

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score

    def _build_custom_scorer(self, constraints: dict):
        recall_min = constraints.get('recall_min', 0.90)
        f1_min = constraints.get('f1_min', 0.70)
        precision_min = constraints.get('precision_min', 0.80)
        threshold = constraints.get('threshold_provisoire', 0.50)

        def custom_score(y_true, y_proba):
            y_pred = (y_proba >= threshold).astype(int)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)

            if recall < recall_min:    return 0.0
            if f1 < f1_min:            return 0.0
            if precision < precision_min: return 0.0

            return f1  # on maximise f1 parmi les combinaisons valides

        return make_scorer(custom_score, needs_proba=True)

    @abstractmethod
    def _run_grid_search(self, X_train, y_train):
        raise NotImplementedError

    def threshold(self, y_data, y_proba):
        threshold_config = self.config['evaluation']['threshold']

        thresholds = np.arange(
            threshold_config['search']['start'],
            threshold_config['search']['stop'],
            threshold_config['search']['step']
        )

        constraints = self.config['model']['grid_search']['constraints']
        recall_min = constraints['recall_min']
        f1_min = constraints['f1_min']
        #precision_min = constraints['precision_min']

        best_f1 = -1
        chosen_threshold = 0
        predicted_new = np.zeros_like(y_data)
        best_recall = 0
        best_precision = 0

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            r = recall_score(y_data, y_pred, zero_division=0)
            p = precision_score(y_data, y_pred, zero_division=0)
            f1 = f1_score(y_data, y_pred, zero_division=0)

            if r >= recall_min and f1 > best_f1:
                best_f1 = f1
                chosen_threshold = t
                predicted_new = y_pred
                best_recall = r
                best_precision = p

        self._chosen_threshold = chosen_threshold
        self._best_recall = best_recall
        self._best_precision = best_precision
        self._best_f1 = best_f1

        return {
            'recall': best_recall,
            'precision': best_precision,
            'f1': best_f1,
            'pred': predicted_new,
            'threshold': chosen_threshold
        }

    def apply_threshold(self, y_data, y_proba):
        """
        Applique le seuil figé issu du train/validation.
        Aucune recherche ici - le seuil est une contrainte fixe.
        Les métriques sont recalculées sur les vraies prédictions.
        """
        threshold = self.config['evaluation']['threshold_result']['threshold']

        predicted_new = (y_proba >= threshold).astype(int)

        recall = recall_score(y_data, predicted_new, zero_division=0)
        precision = precision_score(y_data, predicted_new, zero_division=0)
        f1 = f1_score(y_data, predicted_new, zero_division=0)

        return recall, precision, f1, predicted_new, threshold