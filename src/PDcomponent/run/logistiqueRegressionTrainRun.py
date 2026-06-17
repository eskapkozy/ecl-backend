from src.pipelines.Features.featurePipeline import FeaturePipeline
from src.Utile.artifactManager import ArtifactType
from src.runAbstraction import RunAbstraction

import numpy as np
import yaml
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, accuracy_score, \
    RocCurveDisplay, PrecisionRecallDisplay

from src.PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from src.PDcomponent.run.pdRUN import PDRun

class LogistiqueRegressionTrainRun(PDRun):

    def __init__(self, train_map: dict = None, test_map: dict = None, val_map: dict = None,config_path: str = None):
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
            # model param log
            mlflow.log_params(self.config['model'])

            # model fit + get artefact
            model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, penalty=penalty, solver=solver,
                                       C=C, tol=tol)
            self.model_artifact = model.fit(x_train_resampled, y_train_resampled)

            # ########################
            # Validation Prediction
            # #######################

            y_predict = model.predict(x_val_transformed)
            y_proba = model.predict_proba(x_val_transformed)[:, 1]

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

        return None




    def _run_test(self):


        # #################
        # Load Artifact
        # #################

        binning_process, model_fit = self._load_data()

        # ########################
        # Train data transformation
        # #######################

        self.featurePipeline = FeaturePipeline(self._x_test, self._y_test, config=self.config,binning_process=binning_process)

        x_transformed = self.featurePipeline.transformed
        y_test = self._y_test

        # ########################
        # Model Parameter
        # #######################

        # max_iter = int(self.configs['model']['max_iter'])
        # class_weight = self.configs['model']['class_weight']

        #hyperparameters = self.configs['model']['hyperparameters']

        # ########################
        # Run
        # #######################

        with mlflow.start_run(run_name=self.config['run']['name']):

            # model param log
            mlflow.log_params(self.config['model'])


            # ############
            # Test Prediction
            # ############

            y_pred = model_fit.predict(x_transformed)
            y_prob = model_fit.predict_proba(x_transformed)[:,1]


            # ############
            # Test Metrics  ( F1 - RECALL - ROC - AUC - GINI
            # ############

            # Est-ce que le model discrimine bien ?
            roc_auc = roc_auc_score(y_test, y_prob)
            gini = 2*roc_auc - 1



            # Appliquer le seuil est metrique trouver en train
            recall , precision , f1 , predicted_new , threshold = self.apply_threshold(y_test, y_prob)

            confusion_mtx = confusion_matrix(y_test, predicted_new)
            tn, fp, fn, tp = confusion_mtx.ravel()

            accuracy = accuracy_score(y_test, predicted_new)





            # ########################
            #  Log and Persiste
            # #######################

            # metric log
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

            self._log_roc_fig(y_test, y_prob)
            self._log_precision_recall_fig(y_test, y_prob)

        return None








    # ########################
    #  Persiste
    # #######################






    def _load_data(self):

        run_id = self.config['mlflow']['run_artifact_path']['run_id']
        binning_config = self._run_artifact_path['binning_process']
        model_fit_config = self._run_artifact_path['model_fit']

        config = [binning_config, model_fit_config]

        return self._artifactmanager.load_All(run_id = run_id,configList=config)






