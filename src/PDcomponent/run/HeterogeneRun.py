import copy

import numpy as np
import matplotlib.pyplot as plt
import mlflow

import optuna
from sklearn import ensemble

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

    def __init__(self, train_map: dict = None, test_map: dict = None, val_map: dict = None, config_path: str = None,test_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path,test_path)

    def _run_train(self):
        # ########################
        # Train data transformation
        # ########################

        self.featurePipeline = PDFeaturePipeline(window_months=12, woe_config=self.config['woe'])

        self.x_train_resampled, self.y_train_resampled = self.featurePipeline.apply_woe(self._x_train, self._y_train)
        binning_process = self.featurePipeline.binning_process

        # ########################
        # Validation data transformation — pipeline dédié, pas réutilisation
        # ########################

        y_val = self._y_val

        pipeline_val = PDFeaturePipeline(
            window_months=12,
            woe_config=self.config['woe'],
            binning_process=binning_process
        )
        x_val_transformed, _ = pipeline_val.apply_woe(self._x_val)

        # ########################
        # Model Parameters — tout vient de self.config
        # ########################

        run_params = self.config['run']


        # class weight — calculé dynamiquement à partir des données resampled
        neg = (self.y_train_resampled == 0).sum()
        pos = (self.y_train_resampled == 1).sum()
        scale_pos_weight = neg / pos

        # ########################
        # Run
        # ########################

        with mlflow.start_run(run_name=run_params['name']) as run:
            # mlflow.log_param('random_state', random_state)
            # mlflow.log_param('scale_pos_weight', scale_pos_weight)

            optimal_fit = self._run_grid_search(self.x_train_resampled, self.y_train_resampled, x_val_transformed,
                                                y_val)
            self._model_artifact = optimal_fit['best_estimator']

            # ########################
            # Validation Prediction
            # ########################

            y_proba = self._model_artifact.predict_proba(x_val_transformed)[:, 1]

            # ########################
            # Metrics
            # ########################

            roc_auc = roc_auc_score(y_val, y_proba)
            gini = 2 * roc_auc - 1

            handeler = self.threshold(y_val, y_proba)
            predicted_new = handeler['pred']

            confusion_mtx = confusion_matrix(y_val, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()
            accuracy = accuracy_score(y_val, predicted_new)

            recall = handeler['recall']
            precision = handeler['precision']
            f1 = handeler['f1']

            # ########################
            # Log and Persist
            # ########################

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

            self._log_model_artifact(self._model_artifact)
            self._log_feature_artifact(self.featurePipeline)
            self._log_roc_curve(y_data=y_val, y_proba=y_proba)
            self._log_precision_recall_curve(y_data=y_val, y_proba=y_proba)

            # log model hyerparamettre
            # mlflow.log_params(self.config['model'])
            mlflow.log_params(optimal_fit['best_params'])

            if self.test_path is not None:
                self.save_evaluation_metrics(self.test_path)

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

    def _run_grid_search(self, x_train, y_train, x_val, y_val) -> dict:
        grid_cfg = self.config['model']['grid_search']
        constraints = grid_cfg['constraints']
        param_space = grid_cfg['param_space']
        n_trials = grid_cfg.get('n_trials', 15)

        def objective(trial):
            params = {
                "lgbm": {
                    "n_estimators": trial.suggest_int("lgbm_n_estimators", *param_space['lgbm']['n_estimators']),
                    "learning_rate": trial.suggest_float("lgbm_learning_rate", *param_space['lgbm']['learning_rate'],
                                                         log=True),
                    "num_leaves": trial.suggest_int("lgbm_num_leaves", *param_space['lgbm']['num_leaves']),
                },
                "xgb": {
                    "n_estimators": trial.suggest_int("xgb_n_estimators", *param_space['xgb']['n_estimators']),
                    "learning_rate": trial.suggest_float("xgb_learning_rate", *param_space['xgb']['learning_rate'],
                                                         log=True),
                    "max_depth": trial.suggest_int("xgb_max_depth", *param_space['xgb']['max_depth']),
                },
                "meta": {
                    "C": trial.suggest_float("meta_C", *param_space['meta']['C'], log=True),
                }
            }

            # Merge avec le config de base (subsample/colsample fixes, non tunés)
            trial_config = copy.deepcopy(self.config)
            trial_config["model"]["hyperparameters"]["lgbm"].update(params["lgbm"])
            trial_config["model"]["hyperparameters"]["xgb"].update(params["xgb"])
            trial_config["model"]["hyperparameters"]["meta"]["C"] = params["meta"]["C"]

            # Build + fit
            ensemble = self._build_ensemble(y_train, params=trial_config["model"]["hyperparameters"])
            ensemble.fit(x_train, y_train)

            # Évaluation sur validation
            y_proba = ensemble.predict_proba(x_val)[:, 1]
            result = self.threshold(y_val, y_proba)

            r = result['recall']
            f1 = result['f1']

            if r < constraints['recall_min'] and f1 < constraints['f1_min']:
                return 0.0

            return f1

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        hp = self.config["model"]["hyperparameters"]

        best_hp= {
            "lgbm": {
                "n_estimators": best["lgbm_n_estimators"],
                "learning_rate": best["lgbm_learning_rate"],
                "num_leaves": best["lgbm_num_leaves"],
                "subsample": hp["lgbm"]["subsample"],
                "colsample_bytree": hp["lgbm"]["colsample_bytree"],
            },
            "xgb": {
                "n_estimators": best["xgb_n_estimators"],
                "learning_rate": best["xgb_learning_rate"],
                "max_depth": best["xgb_max_depth"],
                "subsample": hp["xgb"]["subsample"],
                "colsample_bytree": hp["xgb"]["colsample_bytree"],
            },
            "rf": hp["rf"],
            "lr": hp["lr"],
            "svm": hp["svm"],
            "meta": {
                "C": best["meta_C"],
            },
            "stacking_cv_folds": hp.get("stacking_cv_folds", 5),
            "passthrough": hp.get("passthrough", False),
            "calibration_cv": hp.get("calibration_cv", 5),
        }


        return {
            "best_estimator": ensemble,
            "best_params": best_hp
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
                'threshold': threshold,
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





















