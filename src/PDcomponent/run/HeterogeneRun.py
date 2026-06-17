import copy

import numpy as np
import matplotlib.pyplot as plt
import mlflow

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

from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from src.PDcomponent.run.pdRUN import PDRun


class Heterogene(PDRun):

    def __init__(self, train_map: dict = None, test_map: dict = None, val_map: dict = None, config_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path)

    def _run_train(self):
        # ########################
        # Train data transformation
        # #######################

        self.featurePipeline = PDFeaturePipeline(window_months=12, woe_config=self.config['woe'])

        x_train_resampled, y_train_resampled = self.featurePipeline.apply_woe(self._x_train, self._y_train)
        binning_process = self.featurePipeline.binning_process

        # ########################
        # Validation data transformation
        # #######################

        y_val = self._y_val

        pipeline_val = PDFeaturePipeline(window_months=12, woe_config=self.config['woe'],
                                         binning_process=binning_process)
        x_val_transformed, _ = self.featurePipeline.apply_woe(self._x_val)

        # ########################
        # Model Parametter
        # #######################

        # Tuning Optuna avant le fit final
        best_params = self._tune(x_train_resampled, y_train_resampled, x_val_transformed, y_val)

        # ─── 3. Construire l'ensemble ─────────────────────────────────────────

        ensemble_pipeline = self._build_ensemble(y_train_resampled, params=best_params)





        # ########################
        # Run
        # #######################

        with mlflow.start_run(run_name=self.config['run']['name']) as run:
            # model param log
            mlflow.log_params(self.config['model'])

            # Fit — SMOTE est encapsulé dans le pipeline (appliqué par fold)
            ensemble_pipeline.fit(x_train_resampled, y_train_resampled)
            self.model_artifact = ensemble_pipeline

            # Log optuna ici
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items() if not isinstance(v, dict)})


            # ########################
            # Validation Prediction
            # #######################

            # ─── 5. Prédiction validation ─────────────────────────────────

            y_proba = ensemble_pipeline.predict_proba(x_val_transformed)[:, 1]
            print(np.percentile(y_proba, [10, 25, 50, 75, 90]))

            # ########################
            # Metric  ( F1 - RECALL - ROC - AUC - GINI
            # #######################

            # est-ce que le model discrimine bien ?
            roc_auc = roc_auc_score(y_val, y_proba)
            gini = 2 * roc_auc - 1

            # On définit le seuil suivant les contraintes metier
            handeler = self.threshold(y_val, y_proba)
            best_recall, best_precision, best_f1, predicted_new, chosen_threshold = handeler


            predicted_new = handeler['pred']

            confusion_mtx = confusion_matrix(y_val, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()

            accuracy = accuracy_score(y_val, predicted_new)

            recall = handeler['recall']

            precision = handeler['precision']

            f1 = handeler['f1']

            # preparer le save des metrique d'evaluation
            self._setEvaluationMetrics(
                chosen_threshold=chosen_threshold, best_recall=best_recall,
                best_precision=best_precision, best_f1=best_f1)

            # ########################
            #  Log and Persiste
            # #######################

            # metric log
            mlflow.log_metrics({
                'chosen_threshold': handeler['threshold'],
                'roc_auc': roc_auc,
                'gini': gini,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp

            })

            # model artefact  + pipline report
            self._log_model_artifact(self._model_artifact)
            self._log_feature_artifact(self.featurePipeline)
            self._log_roc_curve(y_data=y_val, y_proba=y_proba)
            self._log_precision_recall_curve(y_data=y_val, y_proba=y_proba)

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
            ("lgbm", CalibratedClassifierCV(lgbm, method="isotonic", cv=n_cv_calib)),
            ("xgb", CalibratedClassifierCV(xgb_clf, method="isotonic", cv=n_cv_calib)),
            ("rf", CalibratedClassifierCV(rf, method="isotonic", cv=n_cv_calib)),
            ("lr", lr_pipeline),  # LogReg est nativement calibré
            ("svm", svm_pipeline),
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

            trial_config = copy.deepcopy(self.config)
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
                flat.update(Heterogene._flatten_params(v, prefix=key))
            else:
                flat[key] = v
        return flat

    # ###########################################################################################################
    # Test
    # ######################################################################################################





    def _run_test(self):
        # ########################
        # Load Artifact
        # ########################

        binning_process, model_fit = self._load_data()

        # ########################
        # Test data transformation
        # ########################

        self.featurePipeline = PDFeaturePipeline(
            window_months=12,
            woe_config=self.config['woe'],
            binning_process=binning_process
        )

        x_transformed, _ = self.featurePipeline.apply_woe(self._x_test)
        y_test = self._y_test

        # ########################
        # Run
        # ########################

        with mlflow.start_run(run_name=self.config['run']['name']):
            mlflow.log_params(self.config['model'])

            y_prob = model_fit.predict_proba(x_transformed)[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob)
            gini = 2 * roc_auc - 1

            recall, precision, f1, predicted_new, threshold = self.apply_threshold(y_test, y_prob)

            confusion_mtx = confusion_matrix(y_test, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()
            accuracy = accuracy_score(y_test, predicted_new)

            mlflow.log_metrics({
                'roc_auc': roc_auc,
                'gini': gini,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            })

            self._log_roc_curve(y_test, y_prob)
            self._log_precision_recall_curve(y_test, y_prob)

        return None

    def _load_data(self):
        run_id = self.config['mlflow']['run_artifact_path']['run_id']
        binning_config = self.config['mlflow']['run_artifact_path']['binning_process']
        model_fit_config = self.config['mlflow']['run_artifact_path']['model_fit']

        config = [binning_config, model_fit_config]
        return self._artifact_manager.load_All(run_id=run_id, configList=config)





















