import optuna
from sklearn.model_selection import GridSearchCV


import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import  roc_auc_score, confusion_matrix, accuracy_score


from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from src.PDcomponent.run.pdRUN import PDRun

class LogistiqueRegressionTrainRun(PDRun):

    def __init__(self, train_map: dict = None, test_map: dict = None, val_map: dict = None,config_path: str = None,test_path: str = None):
        super().__init__(train_map, test_map, val_map, config_path,test_path)



    def _run_train(self):
        # ########################
        # Train data transformation
        # #######################

        self.featurePipeline = PDFeaturePipeline(window_months=12, woe_config=self.config['woe'])

        self.x_train_resampled, self.y_train_resampled = self.featurePipeline.apply_woe(self._x_train, self._y_train)
        binning_process = self.featurePipeline.binning_process



        # ########################
        # Validation data transformation
        # #######################

        y_val = self._y_val

        pipeline_val= PDFeaturePipeline(window_months=12, woe_config=self.config['woe'], binning_process=binning_process)
        x_val_transformed, _ = self.featurePipeline.apply_woe(self._x_val)



        # ########################
        # Model Parametter
        # #######################

        hyperparameters = self.config['model']['hyperparameters']

        # class weight
        class_weight = hyperparameters['class_weight']

        # nombre max d'itération
        max_iter = int(hyperparameters['max_iter'])

        # Algo d'optimisation
        solver = hyperparameters['solver']

        # regularisation
        penalty = hyperparameters['regularisation']['penalty']

        # tolerance
        tol = hyperparameters['regularisation']['tol']

        C = hyperparameters['regularisation']['C']
        # random_state = hyperparameters['regularisation']['random_state']

        # ########################
        # Run
        # #######################

        with mlflow.start_run(run_name=self.config['run']['name']) as run:



            # model fit + get artefact

            optimal_fit = self._run_grid_search(self.x_train_resampled, self.y_train_resampled,x_val_transformed,y_val)
            self._model_artifact = optimal_fit['best_estimator']




            #model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, penalty=penalty, solver=solver, C=C, tol=tol)
            #self._model_artifact = model.fit(x_train_resampled, y_train_resampled)

            # ########################
            # Validation Prediction
            # #######################

            y_predict = self._model_artifact.predict(x_val_transformed)
            y_proba = self._model_artifact.predict_proba(x_val_transformed)[:, 1]

            # ########################
            # Metric  ( F1 - RECALL - ROC - AUC - GINI
            # #######################

            # est-ce que le model discrimine bien ?
            roc_auc = roc_auc_score(y_val, y_proba)
            gini = 2 * roc_auc - 1

            # On définit le seuil suivant les contraintes metier
            handeler = self.threshold(y_val, y_proba)
            print(handeler)

            predicted_new = handeler['pred']

            confusion_mtx = confusion_matrix(y_val, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()

            accuracy = accuracy_score(y_val, predicted_new)

            recall = handeler['recall']

            precision = handeler['precision']

            f1 = handeler['f1']

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
            # model param log
            mlflow.log_params(optimal_fit['best_params'])

            if self.test_path is not None:
                self.save_evaluation_metrics(self.test_path)

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
        run_id              = self.config['mlflow']['run_artifact_path']['run_id']
        binning_config      = self.config['mlflow']['run_artifact_path']['binning_process']
        model_fit_config    = self.config['mlflow']['run_artifact_path']['model_fit']

        config = [binning_config, model_fit_config]
        return self._artifact_manager.load_All(run_id=run_id, configList=config)




    def _run_grid_search(self, X_train, y_train, X_val, y_val):
        grid_cfg = self.config['model']['grid_search']
        constraints = grid_cfg['constraints']
        param_space = grid_cfg['param_space']
        n_trials = grid_cfg.get('n_trials', 15)

        def objective(trial):
            params = {
                'C': trial.suggest_float('C', *param_space['C'], log=True),
                'max_iter': trial.suggest_int('max_iter', *param_space['max_iter']),
                'penalty': trial.suggest_categorical('penalty', param_space['penalty']),
                'solver': trial.suggest_categorical('solver', param_space['solver']),
            }

            model = LogisticRegression(
                **params,
                class_weight=self.config['model']['hyperparameters']['class_weight'],
                tol=self.config['model']['hyperparameters']['regularisation']['tol'],
                random_state=self.config['run']['random_state'],
            )

            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_val)[:, 1]
            result = self.threshold(y_val, y_proba)

            r = result['recall']
            f1 = result['f1']

            print(roc_auc_score(y_val, y_proba))
            print(r)

            if r < constraints['recall_min'] and f1 < constraints['f1_min']:
                return 0.0
            return f1

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_model = LogisticRegression(
            **study.best_params,
            class_weight=self.config['model']['hyperparameters']['class_weight'],
            tol=self.config['model']['hyperparameters']['regularisation']['tol'],
            random_state=self.config['run']['random_state'],
        )
        best_model.fit(X_train, y_train)

        return {
            'best_estimator': best_model,
            'best_params': study.best_params,
        }















