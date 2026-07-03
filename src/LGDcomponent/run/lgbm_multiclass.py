import mlflow
import numpy as np
import optuna
import pandas as pd

from src.LGDcomponent.run.LGD_run import LGDRun

import lightgbm as lgb
from lightgbm import LGBMClassifier


class LgbmMulticlass(LGDRun):

    def __init__(self, train_map: dict = None, test_map: dict = None, val_map: dict = None,config_path: str = None,test_path:str =None):
        super().__init__(train_map, test_map, val_map, config_path,test_path)

        self.y_proba = None


    def _run_train(self):
        # ########################
        # Train — discrétisation
        # ########################
        # fit sur train uniquement, transform sur val séparément
        discretizer = self._fit_discretizer(pd.Series(self._y_train), n_bins=self.config['model']['n_bins'])
        y_train_bins = discretizer.transform(pd.Series(self._y_train))
        y_val_bins = self._transform_with_discretizer(pd.Series(self._y_val),discretizer=discretizer)

        # ########################
        # Validation — data transformation
        # ########################
        x_train = self._x_train
        x_val = self._x_val

        # ########################
        # Run
        # ########################
        run_params = self.config['run']

        with mlflow.start_run(run_name=run_params['name']) as run:
            optimal_fit = self._run_grid_search(x_train, y_train_bins, x_val, y_val_bins)
            self._model_artifact = optimal_fit['best_estimator']

            # ########################
            # Validation Prediction
            # ########################
            proba = self._model_artifact.predict_proba(x_val)

            # ########################
            # Metrics
            # ########################
            metrics = self._evaluate(
                y_true_continuous=self._y_val.values,
                y_true_bins=np.array(y_val_bins.values),
                proba=proba
            )

            # ########################
            # Log and Persist
            # ########################
            mlflow.log_metrics({
                'rmse': metrics['rmse'],
                'dxy': metrics['dxy'],
                'ece': metrics['ece'],
                'multiclass_accuracy': metrics['multiclass_accuracy'],
                'multiclass_log_loss': metrics['multiclass_log_loss'],
            })

            mlflow.log_params(optimal_fit['best_params'])

            self._log_model_artifact(self._model_artifact)
            self._log_feature_artifact()  # discretizer + bin_summary
            self._log_calibration_by_bin(y_val_bins.values, proba)

            if self.test_path is not None:
                self.save_evaluation_metrics(self.test_path)




    def _run_test(self):
        # ########################
        # Load Artifacts
        # ########################
        discretizer, model_fit = self._load_data()
        self.discretizer = discretizer

        # ########################
        # Discrétisation test
        # ########################
        y_test_bins = self._transform_with_discretizer(self._y_test)

        # ########################
        # Run
        # ########################
        with mlflow.start_run(run_name=self.config['run']['name']):
            proba = model_fit.predict_proba(self._x_test)

            metrics = self._evaluate(
                y_true_continuous=self._y_test.values,
                y_true_bins=y_test_bins.values,
                proba=proba
            )

            mlflow.log_metrics({
                'rmse': metrics['rmse'],
                'dxy': metrics['dxy'],
                'ece': metrics['ece'],
                'multiclass_accuracy': metrics['multiclass_accuracy'],
                'multiclass_log_loss': metrics['multiclass_log_loss'],
            })

            self._log_calibration_by_bin(y_test_bins.values, proba)
            mlflow.log_params(self.config['model'])

            self.y_pred = metrics['y_pred']
            self.y_proba = proba

        return None



    def _run_grid_search(self, X_train, y_train, X_val, y_val):


        grid_cfg = self.config['model']['grid_search']
        param_space = grid_cfg['param_space']
        n_trials = grid_cfg.get('n_trials', 20)
        n_bins = self.discretizer.n_bins

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', *param_space['max_depth']),
                'num_leaves': trial.suggest_int('num_leaves', *param_space['num_leaves']),
                'min_child_samples': trial.suggest_int('min_child_samples', *param_space['min_child_samples']),
                'reg_lambda': trial.suggest_float('reg_lambda', *param_space['reg_lambda']),
                'reg_alpha': trial.suggest_float('reg_alpha', *param_space['reg_alpha']),
                'subsample': trial.suggest_float('subsample', *param_space['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *param_space['colsample_bytree']),
                'learning_rate': trial.suggest_float('learning_rate', *param_space['learning_rate']),
                'n_estimators': trial.suggest_int('n_estimators', *param_space['n_estimators']),
            }

            model = LGBMClassifier(
                **params,
                objective='multiclass',
                num_class=n_bins,
                random_state=self.config['run']['random_state'],
                verbose=-1,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(
                    self.config['model']['hyperparameters']['early_stopping_rounds'],
                    verbose=False
                )],
            )

            proba = model.predict_proba(X_val)
            metrics = self._evaluate(
                y_true_continuous=self._y_val.values,
                y_true_bins=y_val,
                proba=proba
            )

            print(f"RMSE: {metrics['rmse']:.4f} | Dxy: {metrics['dxy']:.4f} | ECE: {metrics['ece']:.4f}")
            return -metrics['rmse']  # Optuna minimise → on minimise RMSE

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Refit avec les meilleurs params

        best_model = LGBMClassifier(
            **study.best_params,
            objective='multiclass',
            num_class=n_bins,
            random_state=self.config['run']['random_state'],
            verbose=-1,
        )

        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(
                self.config['model']['hyperparameters']['early_stopping_rounds'],
                verbose=False
            )],
        )

        return {
            'best_estimator': best_model,
            'best_params': study.best_params,
        }

    def _load_data(self):
        run_id = self.config['mlflow']['run_artifact_path']['run_id']
        discretizer_config = self.config['mlflow']['run_artifact_path']['lgd_discretizer']
        model_fit_config = self.config['mlflow']['run_artifact_path']['model_fit']

        discretizer, model_fit = self._artifact_manager.load_All(
            run_id=run_id,
            configList=[discretizer_config, model_fit_config]
        )
        return discretizer, model_fit

    def _build_custom_scorer(self, constraints: dict):
        # Pas de scorer custom binaire ici — optimisation directe sur RMSE
        # dans _run_grid_search via Optuna
        pass

    # 1. ADD MISSING ABSTRACT METHOD

    def setup(self):
        """
        Implements RunAbstraction.setup()
        Add initialization pipeline steps here if needed.
        """
        pass

    # 2. ADD MISSING ABSTRACT METHOD

    def _load_scaler(self):
        """
        Implements RunAbstraction._load_scaler()
        LGD modeling might not require a scaler, but the contract demands it.
        """
        pass

