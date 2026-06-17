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
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, recall_score, precision_score, f1_score

from src.runAbstraction import RunAbstraction
from src.Utile.artifactManager import ArtifactType
from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline


class PDRun(RunAbstraction):

    def __init__(self, train_map: dict = None, test_map: dict = None,
                 val_map: dict = None, config_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path)

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
        """
        Écrit les métriques du dernier run dans le fichier de config test.
        À appeler manuellement après validation visuelle des métriques dans MLflow.

        Usage :
            run.run()
            # → vérifier les métriques dans MLflow
            run.save_pd_evaluation_metrics("configs/test_config.yaml")
        """
        path = Path(test_config_path)
        with open(path, "r") as f:
            test_config = yaml.safe_load(f)

        test_config["evaluation"] = {
            "roc_auc"             : True,
            "recall"              : True,
            "precision"           : True,
            "f1_score"            : True,
            "confusion_matrix"    : True,
            "threshold"           : float(self._chosen_threshold),
            "recall_threshold"    : float(self._best_recall),
            "precision_threshold" : float(self._best_precision),
            "f1_threshold"        : float(self._best_f1),
        }

        with open(path, "w") as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)

        print(f"Métriques sauvegardées dans {path}")




    def threshold(self, y_data, y_proba):
        threshold = np.arange(0.1, 0.99, 0.01)
        best_recall = 0
        chosen_threshold = 0
        predicted_new = np.zeros_like(y_data)
        best_f1 = -1
        best_precision = 0

        for t in threshold:
            y_pred = (y_proba >= t).astype(int)
            r = recall_score(y_data, y_pred)
            p = precision_score(y_data, y_pred)
            f1 = f1_score(y_data, y_pred)

            if r >= 0.90 and f1 > best_f1:
                best_f1 = f1
                chosen_threshold = t
                predicted_new = y_pred
                best_recall = r
                best_precision = p

        self._chosen_threshold = chosen_threshold
        self._best_recall = best_recall
        self._best_precision = best_precision
        self._best_f1 = best_f1

        return {'recall': best_recall, "precision": best_precision, "f1": best_f1,
                "pred": predicted_new, "threshold": chosen_threshold}




    def apply_threshold(self, y_data, y_proba):

        """
            Appliquer le seuil est metrique trouver en train
        """

        threshold = self.config['evaluation']['threshold']
        recall = self.config['evaluation']['recall_threshold']
        precision = self.config['evaluation']['precision_threshold']
        f1 = self.config['evaluation']['f1_threshold']

        predicted_new = (y_proba >= threshold).astype(int)

        return recall, precision, f1, predicted_new, threshold