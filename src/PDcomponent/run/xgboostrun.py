
import optuna
from sklearn.calibration import CalibratedClassifierCV


import mlflow
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from src.PDcomponent.run.pdRUN import PDRun


class XGBoostRun(PDRun):

    def __init__(self, train_map: dict = None, test_map: dict = None, val_map: dict = None,config_path: str = None,test_path:str =None):
        super().__init__(train_map, test_map, val_map, config_path,test_path)



    def _run_train(self):
        # ########################
        # Train data transformation
        # ########################

        # Le setup a instaurer le feature pipeline, contenant les scaler artifact
        # les donnee son dja sclaer avant d'applique woe

        self.featurePipeline = PDFeaturePipeline(window_months=12, woe_config=self.config['woe'])

        self.x_train_resampled, self.y_train_resampled = self.featurePipeline.apply_woe(self._x_train, self._y_train)
        binning_process = self.featurePipeline.binning_process



        selector = self.featurePipeline.selector

        # ########################
        # Validation data transformation — pipeline dédié, pas réutilisation
        # ########################

        y_val = self._y_val




        pipeline_val = PDFeaturePipeline(
            window_months=12,
            woe_config=self.config['woe'],
            binning_process=binning_process # binnning appliquer au scale du train,
        )
        x_val_transformed, _ = pipeline_val.apply_woe(self._x_val)

        # ########################
        # Model Parameters — tout vient de self.config
        # ########################

        run_params       = self.config['run']


        # class weight — calculé dynamiquement à partir des données resampled
        neg = (self.y_train_resampled == 0).sum()
        pos = (self.y_train_resampled == 1).sum()
        scale_pos_weight = neg / pos

        # ########################
        # Run
        # ########################

        with mlflow.start_run(run_name=run_params['name']) as run:



            optimal_fit = self._run_grid_search(self.x_train_resampled, self.y_train_resampled,x_val_transformed, y_val)
            self._model_artifact = optimal_fit['best_estimator']

            # ########################
            # Calibration
            # ########################

            calibration_method = self.config['model'].get('calibration', {}).get('method', 'isotonic')
            calibration_CV = self.config['model'].get('calibration', {}).get('cv', 'prefit')

            calibrated_model = CalibratedClassifierCV(
                estimator=self._model_artifact,
                method=calibration_method,
                cv=calibration_CV
            )
            calibrated_model.fit(x_val_transformed, y_val)

            self._model_artifact = calibrated_model  # remplace le modèle brut par le modèle calibré


            # ########################
            # Validation Prediction
            # ########################

            y_proba = self._model_artifact.predict_proba(x_val_transformed)[:, 1]

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
                'chosen_percentile': handeler['percentile'],
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

            mlflow.log_params({'calibration_method': calibration_method})

            self._log_model_artifact(self._model_artifact)
            self._log_feature_artifact(self.featurePipeline)
            self._log_roc_curve(y_data=y_val, y_proba=y_proba)
            self._log_precision_recall_curve(y_data=y_val, y_proba=y_proba)
            self._log_calibration_curve(y_data=y_val, y_proba=y_proba)

            # log model hyerparamettre
            #mlflow.log_params(self.config['model'])
            mlflow.log_params(optimal_fit['best_params'])

            if self.test_path is not None:
                self.save_evaluation_metrics(self.test_path)
            self.y_proba = y_proba
        return None



    def _run_test(self):
        # ########################
        # Load Artifact
        # ########################

        binning_process, model_fit = self._load_data()

        # ########################
        # Test data transformation
        # ########################

        #scaler = self._load_scaler()

        # Scale direct, sans passer par le pipeline
        #selector = FeatureSelector()
        #self._x_scaled = selector.transform(self._x_test, scaler)

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


            y_prob = model_fit.predict_proba(x_transformed)[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob)
            gini    = 2 * roc_auc - 1

            recall, precision, f1, predicted_new, threshold = self.apply_threshold(y_test, y_prob)

            confusion_mtx = confusion_matrix(y_test, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()
            accuracy       = accuracy_score(y_test, predicted_new)

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
            mlflow.log_params(self.config['model'])

            self.y_proba = y_prob

        return None



    def _run_grid_search(self, X_train, y_train, X_val, y_val):
        grid_cfg = self.config['model']['grid_search']
        constraints = grid_cfg['constraints']
        param_space = grid_cfg['param_space']
        n_trials = grid_cfg.get('n_trials', 15)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', *param_space['max_depth']),
                'min_child_weight': trial.suggest_int('min_child_weight', *param_space['min_child_weight']),
                'reg_lambda': trial.suggest_float('reg_lambda', *param_space['reg_lambda']),
                'reg_alpha': trial.suggest_float('reg_alpha', *param_space['reg_alpha']),
                'subsample': trial.suggest_float('subsample', *param_space['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *param_space['colsample_bytree']),
                'learning_rate': trial.suggest_float('learning_rate', *param_space['learning_rate']),
                'n_estimators': trial.suggest_int('n_estimators', *param_space['n_estimators']),
            }

            model = XGBClassifier(
                **params,
                random_state=self.config['run']['random_state'],
                eval_metric=self.config['model']['hyperparameters']['eval_metric'],
                early_stopping_rounds=self.config['model']['hyperparameters']['regularisation']['early_stopping_rounds'],
                scale_pos_weight=self.config['model']['hyperparameters']['scale_pos_weight'],
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_proba = model.predict_proba(X_val)[:, 1]
            result = self.threshold(y_val, y_proba)

            r = result['recall']
            p = result['precision']
            f1 = result['f1']

            print('threshold', result['threshold'])
            print('roc_auc_score', roc_auc_score(y_val, y_proba))
            print('recall',r)
            print('f1',f1)
            print('precision',p)




            #if r < constraints['recall_min'] or p < constraints['precision_min'] or f1 < constraints['f1_min']:
             #   return 0.0

            if r < constraints['recall_min'] and f1 < constraints['f1_min']:
                return 0.0
            return f1

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Refit avec les meilleurs params
        best_model = XGBClassifier(
            **study.best_params,
            random_state=self.config['run']['random_state'],
            eval_metric=self.config['model']['hyperparameters']['eval_metric'],
            early_stopping_rounds=self.config['model']['hyperparameters']['regularisation']['early_stopping_rounds'],
            scale_pos_weight=self.config['model']['hyperparameters']['scale_pos_weight'],
        )

        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        return {
            'best_estimator': best_model,
            'best_params': study.best_params,
        }










    def _load_data(self):
        run_id            = self.config['mlflow']['run_artifact_path']['run_id']
        binning_config    = self.config['mlflow']['run_artifact_path']['binning_process']
        model_fit_config  = self.config['mlflow']['run_artifact_path']['model_fit']

        config = [binning_config, model_fit_config]
        return self._artifact_manager.load_All(run_id=run_id, configList=config)