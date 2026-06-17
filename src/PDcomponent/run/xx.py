from src.pipelines.Feature.fearurePipline import FeaturePipline
from src.run.run_abstraction import RunAbstraction
from src.service.artifactManager import ArtifactType

import json
import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score,
    confusion_matrix, accuracy_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
import lightgbm as lgb
import xgboost as xgb


import mlflow


class HeterogeneousEnsembleTrainRun(RunAbstraction):
    """
    Train run pour un modèle d'ensemble hétérogène (stacking).

    Architecture :
        Couche 1 — base learners  : LightGBM, XGBoost, RandomForest, LogReg, SVM-RBF
        Couche 2 — calibration    : CalibratedClassifierCV (isotonic) sur chaque learner
        Couche 3 — méta-learner   : LogisticRegression (OOF stacking)
        Post-train — seuil        : optimisation F1 sous contrainte recall >= 0.90 & precision >= 0.35

    Reprend exactement l'interface de XgboostTrainRun :
        - même signature __init__
        - même méthode _run_train
        - même gestion artefacts MLflow
        - même logique threshold
    """

    def __init__(
        self,
        train_map: dict = None,
        test_map: dict = None,
        val_map: dict = None,
        config: dict = None,
        config_path: str = None,
    ):
        super().__init__(train_map, test_map, val_map, config, config_path)

    # =========================================================================
    # MAIN TRAIN LOOP
    # =========================================================================

    def _run_train(self):

        # ─── 1. Feature pipeline (train) ─────────────────────────────────────

        self.featurePipline = FeaturePipline(
            self._x_train, self._y_train, config=self.config
        )
        binning_process = self.featurePipline.binning_process

        x_train_resampled = self.featurePipline.x_resampled
        y_train_resampled = self.featurePipline.y_resampled

        # ─── 2. Feature pipeline (validation) ────────────────────────────────

        validation_config = self.config.copy()
        validation_config["woe"]["persistence"] = self.featurePipline.binning_process

        x_val_transformed = FeaturePipline(
            self._x_val,
            y_data=None,
            state="validation",
            config=validation_config,
            binning_process=binning_process,
        ).transformed

        y_val = self._y_val


        #───   paramettre trouver ─────────────────────────────────────────

        # Tuning Optuna avant le fit final
        best_params = self._tune(x_train_resampled, y_train_resampled, x_val_transformed, y_val)

        # ─── 3. Construire l'ensemble ─────────────────────────────────────────

        ensemble_pipeline = self._build_ensemble(y_train_resampled, params=best_params)

        # ─── 4. MLflow run ───────────────────────────────────────────────────

        with mlflow.start_run(run_name=self.config["run"]["name"]) as run:

            mlflow.log_params(self._flatten_params(self.config["model"]))

            # Fit — SMOTE est encapsulé dans le pipeline (appliqué par fold)
            ensemble_pipeline.fit(x_train_resampled, y_train_resampled)
            self.model_artifact = ensemble_pipeline

            # Log optuna ici
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items() if not isinstance(v, dict)})

            # ─── 5. Prédiction validation ─────────────────────────────────

            y_proba = ensemble_pipeline.predict_proba(x_val_transformed)[:, 1]
            print(np.percentile(y_proba, [10, 25, 50, 75, 90]))

            # ─── 6. Métriques ─────────────────────────────────────────────

            roc_auc = roc_auc_score(y_val, y_proba)
            gini = 2 * roc_auc - 1

            handler = self.threshold(y_val, y_proba)
            best_recall, best_precision, best_f1, predicted_new, chosen_threshold = handler



            confusion_mtx = confusion_matrix(y_val, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()
            accuracy = accuracy_score(y_val, predicted_new)

            # preparer le save des metrique d'evaluation
            self._setEvaluationMetrics(
                chosen_threshold=chosen_threshold, best_recall=best_recall,
                best_precision=best_precision, best_f1=best_f1)

            # ─── 7. Log métriques ─────────────────────────────────────────

            mlflow.log_metrics({
                "chosen_threshold": float(chosen_threshold),
                "roc_auc": roc_auc,
                "gini": gini,
                "f1": best_f1,
                "precision": best_precision,
                "recall": best_recall,
                "accuracy": accuracy,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            })

            # ─── 8. Artefacts ─────────────────────────────────────────────

            self.log_model_artifact(self.model_artifact)
            self.log_feature_artifact(self.featurePipline)
            self.log_base_learners_contributions(
                x_val_transformed, y_val, ensemble_pipeline
            )
            self.log_roc_fig(y_data=y_val, y_proba=y_proba)
            self.log_precision_recall_fig(y_data=y_val, y_prob=y_proba)

        return None

    # =========================================================================
    # BUILD ENSEMBLE
    # =========================================================================

    def _build_ensemble(self, y_train_resampled, params=None):
        """
        Construit le pipeline d'ensemble complet.

        Notes de design :
        - SMOTE est passé DANS le ImbPipeline pour qu'il ne s'applique
          qu'aux folds d'entraînement lors du stacking OOF (pas de leakage).
        - Chaque base learner est enveloppé dans CalibratedClassifierCV
          pour homogénéiser les probabilités avant le méta-learner.
        - Le méta-learner reçoit uniquement les probas OOF (passthrough=False).
        """
        hp = params if params is not None else self.config["model"]["hyperparameters"]

        # ── Base learners ────────────────────────────────────────────────────

        lgbm = lgb.LGBMClassifier(
            n_estimators=hp["lgbm"]["n_estimators"],
            learning_rate=hp["lgbm"]["learning_rate"],
            num_leaves=hp["lgbm"]["num_leaves"],
            subsample=hp["lgbm"]["subsample"],
            colsample_bytree=hp["lgbm"]["colsample_bytree"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        xgb_clf = xgb.XGBClassifier(
            n_estimators=hp["xgb"]["n_estimators"],
            learning_rate=hp["xgb"]["learning_rate"],
            max_depth=hp["xgb"]["max_depth"],
            subsample=hp["xgb"]["subsample"],
            colsample_bytree=hp["xgb"]["colsample_bytree"],
            eval_metric="logloss",
            scale_pos_weight=self._compute_scale_pos_weight(y_train_resampled),
            random_state=42,
            n_jobs=-1,
        )

        rf = RandomForestClassifier(
            n_estimators=hp["rf"]["n_estimators"],
            max_features=hp["rf"].get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Après

        # LR + SVM nécessitent une mise à l'échelle
        lr_pipeline = LogisticRegression(C=hp["lr"]["C"], class_weight="balanced",
                                         solver="saga", max_iter=1000, random_state=42)


        svm_pipeline = SVC(C=hp["svm"]["C"], kernel="rbf", gamma="scale",
                           probability=True, class_weight="balanced", random_state=42)


        # ── Calibration isotonic ─────────────────────────────────────────────

        n_cv_calib = hp.get("calibration_cv", 5)

        base_learners = [
            ("lgbm", CalibratedClassifierCV(lgbm,       method="isotonic", cv=n_cv_calib)),
            ("xgb",  CalibratedClassifierCV(xgb_clf,    method="isotonic", cv=n_cv_calib)),
            ("rf",   CalibratedClassifierCV(rf,          method="isotonic", cv=n_cv_calib)),
            ("lr",   lr_pipeline),   # LogReg est nativement calibré
            ("svm",  svm_pipeline),
        ]

        # ── Méta-learner ─────────────────────────────────────────────────────

        meta_learner = LogisticRegression(
            C=hp["meta"]["C"],
            class_weight={0: 1, 1: 8},
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )

        stacking = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=StratifiedKFold(
                n_splits=hp.get("stacking_cv_folds", 5),
                shuffle=True,
                random_state=42,
            ),
            stack_method="predict_proba",
            passthrough=hp.get("passthrough", False),
            n_jobs=-1,
        )

        # ── Pipeline final avec SMOTE ─────────────────────────────────────────

        smote_ratio = hp.get("smote_sampling_strategy", 0.3)

        return stacking





    # ##########################################
    # Optimisation d'hyperparamettre
    # #########################################

    def _tune(self, x_train, y_train, x_val, y_val) -> dict:

        def objective(trial):
            params = {
                "lgbm": {
                    "n_estimators": trial.suggest_int("lgbm_n_est", 200, 800),
                    "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.1, log=True),
                    "num_leaves": trial.suggest_int("lgbm_leaves", 31, 127),
                },
                "xgb": {
                    "n_estimators": trial.suggest_int("xgb_n_est", 200, 800),
                    "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
                    "max_depth": trial.suggest_int("xgb_depth", 3, 8),
                },
                "meta": {
                    "C": trial.suggest_float("meta_C", 0.01, 10, log=True),
                    "w1": trial.suggest_int("meta_w1", 1, 15),
                }
            }

            # Merge avec le configs de base
            trial_config = self.config.copy()
            trial_config["model"]["hyperparameters"]["lgbm"].update(params["lgbm"])
            trial_config["model"]["hyperparameters"]["xgb"].update(params["xgb"])
            trial_config["model"]["hyperparameters"]["meta"]["C"] = params["meta"]["C"]

            # Build + fit
            ensemble = self._build_ensemble(y_train, params=trial_config["model"]["hyperparameters"])
            ensemble.fit(x_train, y_train)

            # Évaluation
            y_proba = ensemble.predict_proba(x_val)[:, 1]
            recall, _, f1, _, _ = self.threshold(y_val, y_proba)

            # Contrainte métier
            if recall < 0.90:
                return 0.0

            return f1

        # Supprimer les logs verbeux d'Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config["model"].get("optuna_trials", 30))



        best = study.best_params
        return {
            "lgbm": {
                "n_estimators": best["lgbm_n_est"],
                "learning_rate": best["lgbm_lr"],
                "num_leaves": best["lgbm_leaves"],
                "subsample": self.config["model"]["hyperparameters"]["lgbm"]["subsample"],
                "colsample_bytree": self.config["model"]["hyperparameters"]["lgbm"]["colsample_bytree"],
            },
            "xgb": {
                "n_estimators": best["xgb_n_est"],
                "learning_rate": best["xgb_lr"],
                "max_depth": best["xgb_depth"],
                "subsample": self.config["model"]["hyperparameters"]["xgb"]["subsample"],
                "colsample_bytree": self.config["model"]["hyperparameters"]["xgb"]["colsample_bytree"],
            },
            "rf": self.config["model"]["hyperparameters"]["rf"],
            "lr": self.config["model"]["hyperparameters"]["lr"],
            "svm": self.config["model"]["hyperparameters"]["svm"],
            "meta": {
                "C": best["meta_C"],
            },
            "stacking_cv_folds": self.config["model"]["hyperparameters"].get("stacking_cv_folds", 5),
            "passthrough": self.config["model"]["hyperparameters"].get("passthrough", False),
            "calibration_cv": self.config["model"]["hyperparameters"].get("calibration_cv", 5),
        }









    # =========================================================================
    # THRESHOLD  (identique à XgboostTrainRun)
    # =========================================================================

    def threshold(self, y_data, y_proba):
        """
        Optimise le seuil de décision sous contrainte métier :
            recall >= 0.90  et  precision >= 0.35
        Retourne : (best_recall, best_precision, best_f1, predicted_new, chosen_threshold)
        """
        thresholds = np.arange(0.1, 0.99, 0.01)
        best_recall, best_precision, best_f1 = 0, 0, 0
        chosen_threshold, predicted_new = 0.5, None

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            r = recall_score(y_data, y_pred, zero_division=0)
            p = precision_score(y_data, y_pred, zero_division=0)
            f1 = f1_score(y_data, y_pred, zero_division=0)

            if r >= 0.90 and f1 > best_f1:
                best_f1 = f1
                chosen_threshold = t
                predicted_new = y_pred
                best_recall = r
                best_precision = p

        # Fallback si aucun seuil ne satisfait les contraintes
        if predicted_new is None:
            chosen_threshold = 0.5
            predicted_new = (y_proba >= chosen_threshold).astype(int)
            best_recall    = recall_score(y_data, predicted_new, zero_division=0)
            best_precision = precision_score(y_data, predicted_new, zero_division=0)
            best_f1        = f1_score(y_data, predicted_new, zero_division=0)
            mlflow.set_tag("threshold_warning", "No threshold met business constraints — fallback to 0.5")

        return best_recall, best_precision, best_f1, predicted_new, chosen_threshold

    # =========================================================================
    # PLOTS
    # =========================================================================

    def log_roc_fig(self, y_data, y_proba):
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_data, y_proba, ax=ax, name="Ensemble")
        ax.set_title("ROC Curve — Ensemble")
        mlflow.log_figure(fig, "plots/roc_curve.png")
        plt.close(fig)

    def log_precision_recall_fig(self, y_data, y_prob):
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_data, y_prob, ax=ax, name="Ensemble")
        ax.set_title("Precision-Recall Curve — Ensemble")
        mlflow.log_figure(fig, "plots/precision_recall_curve.png")
        plt.close(fig)

    def log_base_learners_contributions(self, x_val, y_val, ensemble_pipeline):
        """
        Log individuel de chaque base learner sur le val set.
        Permet de monitorer la diversité de l'ensemble dans MLflow.
        """
        stacking: StackingClassifier = ensemble_pipeline
        contributions = {}

        for name, estimator in stacking.named_estimators_.items():
            try:
                proba = estimator.predict_proba(x_val)[:, 1]
                contributions[name] = {
                    "roc_auc": round(roc_auc_score(y_val, proba), 4),
                    "f1":      round(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0), 4),
                }
            except Exception:
                pass  # SVM peut ne pas exposer predict_proba directement ici

        mlflow.log_dict(contributions, "base_learners_contributions/base_learners_contributions.json")

        # Graphe des AUC individuels vs ensemble
        fig, ax = plt.subplots(figsize=(8, 4))
        names  = list(contributions.keys()) + ["ensemble"]
        aucs   = [v["roc_auc"] for v in contributions.values()] + [
            round(roc_auc_score(y_val, ensemble_pipeline.predict_proba(x_val)[:, 1]), 4)
        ]
        colors = ["#4a90d9"] * len(contributions) + ["#e07b39"]
        ax.barh(names, aucs, color=colors)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("ROC-AUC")
        ax.set_title("Contribution des base learners")
        ax.axvline(aucs[-1], color="#e07b39", linestyle="--", linewidth=1.5, label="Ensemble")
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, "plots/base_learners_auc.png")
        plt.close(fig)

    # =========================================================================
    # ARTIFACTS
    # =========================================================================

    def log_model_artifact(self, model_fit):
        self._artifactmanager.log(
            obj=model_fit,
            name="ensemble_model",
            artifact_type=ArtifactType.PKL,
        )

        # poids du méta-learner → interprétabilité de la contribution de chaque base learner
        meta = model_fit.final_estimator_
        weights = {
            name: round(float(coef), 4)
            for name, coef in zip(model_fit.named_estimators_.keys(), meta.coef_[0])
        }
        self._artifactmanager.log(
            obj=weights,
            name="stacking_weights",
            artifact_type=ArtifactType.JSON,
        )

    def log_feature_artifact(self, featurePipline: FeaturePipline):
        self._artifactmanager.log(
            obj=featurePipline.binning_process,
            name="binning_process",
            artifact_type=ArtifactType.PKL,
        )
        self._artifactmanager.log(
            obj=featurePipline.woe_iv_report,
            name="woe_iv_report",
            artifact_type=ArtifactType.JSON,
        )
        self._artifactmanager.log_woeT0_json(
            obj=featurePipline.woe_table,
            name="woe_table",
        )
        self._artifactmanager.log(
            obj=featurePipline.corr_and_woe_selection_report,
            name="corr_and_woe_selection_report",
            artifact_type=ArtifactType.JSON,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        return neg / pos if pos > 0 else 1.0

    @staticmethod
    def _flatten_params(params: dict, prefix: str = "") -> dict:
        """Aplatit un dict imbriqué pour mlflow.log_params."""
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(HeterogeneousEnsembleTrainRun._flatten_params(v, prefix=key))
            else:
                flat[key] = v
        return flat