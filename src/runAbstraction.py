"""
Run Abstraction
================
Contrat de base pour tous les runs d'entraînement/test, quel que soit
le modèle ECL (PD, LGD, EAD) ou l'algorithme utilisé.

Cette classe définit la logique de travail commune :
    - chargement des maps train/val/test (déjà splittées en amont)
    - configuration MLflow
    - logging générique du modèle entraîné
    - sauvegarde des métriques d'évaluation
    - pattern run() → _run_train() ou _run_test()

Ce que cette classe NE connaît PAS (délégué aux sous-classes) :
    - le WoE, le binning_process (spécifique PD → voir PDRun)
    - le chargement effectif des données (_load_data abstraite)
    - l'entraînement effectif d'un algorithme donné
      (_run_train / _run_test abstraites)

Hiérarchie
----------
RunAbstraction (ce fichier)
        ↓
PDRun                          → contrat WoE, binning_process, métriques classification
        ↓
LogisticRegressionRun, XGBoostRun, ...   → implémentation concrète du modèle
"""

from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import yaml

from src.Utile.artifactManager import ArtifactManager, ArtifactType


class RunAbstraction(ABC):

    def __init__(self, train_map: dict = None, test_map: dict = None,
                 val_map: dict = None, config_path: str = None):

        self.config_path = config_path
        self.config       = self._get_config()
        self._is_train    = self.config["run"]["is_train"]

        self._validate_maps(train_map, val_map, test_map)

        self._x_train = self._y_train = None
        self._x_val   = self._y_val   = None
        self._x_test  = self._y_test  = None

        if self._is_train:
            self._x_train = train_map["x_train"]
            self._y_train = train_map["y_train"]
            self._x_val   = val_map["x_val"] if val_map is not None else None
            self._y_val   = val_map["y_val"] if val_map is not None else None
        else:
            self._x_test = test_map["x_test"] if test_map is not None else None
            self._y_test = test_map["y_test"] if test_map is not None else None

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        self._model_artifact = None
        self._artifact_manager = ArtifactManager()


    # ------------------------------------------------------------------
    # Contrat — à implémenter par les sous-classes concrètes
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_data(self):
        """Charge ou prépare les données nécessaires au run (hors maps déjà fournies)."""
        raise NotImplementedError

    @abstractmethod
    def _run_train(self):
        """Logique d'entraînement propre à l'algorithme (régression logistique, XGBoost, ...)."""
        raise NotImplementedError

    @abstractmethod
    def _run_test(self):
        """Logique d'évaluation sur le jeu de test."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Orchestration commune
    # ------------------------------------------------------------------

    def run(self):
        if self._is_train:
            return self._run_train()
        return self._run_test()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _get_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Artefact — modèle (générique, tout algorithme confondu)
    # ------------------------------------------------------------------

    def _log_model_artifact(self, model_fit):
        self._artifact_manager.log(
            obj=model_fit,
            name="model_fit",
            artifact_type=ArtifactType.PKL
        )

    # ------------------------------------------------------------------
    # Validation des maps fournies
    # ------------------------------------------------------------------

    def _validate_maps(self, train_map, val_map, test_map):
        if not self._is_train and train_map is not None:
            raise ValueError("Un mapping de train a été fourni pendant un run de test.")
        if not self._is_train and val_map is not None:
            raise ValueError("Un mapping de validation a été fourni pendant un run de test.")
        if self._is_train and test_map is not None:
            raise ValueError("Un mapping de test a été fourni pendant un run de train.")

    @abstractmethod
    def save_evaluation_metrics(self, test_config_path: str):
        raise NotImplementedError

    def y_true(self):
        return self._y_test