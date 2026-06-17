from src.Utile.artifactManager import ArtifactType
import mlflow
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from src.PDcomponent.run.pdRUN import PDRun


class XGBoostRun(PDRun):

    def __init__(self, train_map: dict = None, test_map: dict = None,
                 val_map: dict = None, config_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path)

    def _run_train(self):
        # ########################
        # Train data transformation
        # ########################

        self.featurePipeline = PDFeaturePipeline(window_months=12, woe_config=self.config['woe'])

        x_train_resampled, y_train_resampled = self.featurePipeline.apply_woe(self._x_train, self._y_train)
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

        run_params       = self.config['run']
        hyperparameters  = self.config['model']['hyperparameters']
        regularisation   = hyperparameters['regularisation']

        random_state           = run_params['random_state']
        early_stopping_rounds  = hyperparameters['early_stopping_rounds']
        eval_metric            = hyperparameters['eval_metric']
        n_estimators            = hyperparameters['n_estimators']

        # class weight — calculé dynamiquement à partir des données resampled
        neg = (y_train_resampled == 0).sum()
        pos = (y_train_resampled == 1).sum()
        scale_pos_weight = neg / pos

        # ########################
        # Run
        # ########################

        with mlflow.start_run(run_name=run_params['name']) as run:
            mlflow.log_params(self.config['model'])
            mlflow.log_param('random_state', random_state)
            mlflow.log_param('scale_pos_weight', scale_pos_weight)

            model = XGBClassifier(
                # Reproductibilité
                random_state=random_state,

                # Régularisation
                max_depth=regularisation['max_depth'],
                min_child_weight=regularisation['min_child_weight'],
                reg_lambda=regularisation['reg_lambda'],
                reg_alpha=regularisation['reg_alpha'],
                subsample=regularisation['subsample'],
                colsample_bytree=regularisation['colsample_bytree'],
                learning_rate=regularisation['learning_rate'],

                # Arrêt anticipé
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,

                # Métrique de suivi
                eval_metric=eval_metric,

                # Pondération des classes
                scale_pos_weight=scale_pos_weight,
            )

            # eval_set requis pour early_stopping_rounds — calculé AVANT le fit
            self._model_artifact = model.fit(
                x_train_resampled, y_train_resampled,
                eval_set=[(x_val_transformed, y_val)],
                verbose=False
            )

            # ########################
            # Validation Prediction
            # ########################

            y_proba = model.predict_proba(x_val_transformed)[:, 1]

            # ########################
            # Metrics
            # ########################

            roc_auc = roc_auc_score(y_val, y_proba)
            gini    = 2 * roc_auc - 1

            handeler      = self.threshold(y_val, y_proba)
            predicted_new = handeler['pred']

            confusion_mtx       = confusion_matrix(y_val, predicted_new)
            tn, fp, fn, tp       = confusion_mtx.ravel()
            accuracy             = accuracy_score(y_val, predicted_new)

            recall    = handeler['recall']
            precision = handeler['precision']
            f1        = handeler['f1']

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

        return None

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
            gini    = 2 * roc_auc - 1

            recall, precision, f1, predicted_new, threshold = self.apply_threshold(y_test, y_prob)

            confusion_mtx = confusion_matrix(y_test, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()
            accuracy       = accuracy_score(y_test, predicted_new)

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
        run_id            = self.config['mlflow']['run_artifact_path']['run_id']
        binning_config    = self.config['mlflow']['run_artifact_path']['binning_process']
        model_fit_config  = self.config['mlflow']['run_artifact_path']['model_fit']

        config = [binning_config, model_fit_config]
        return self._artifact_manager.load_All(run_id=run_id, configList=config)