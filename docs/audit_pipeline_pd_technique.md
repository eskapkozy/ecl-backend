# Audit Technique — Pipeline PD (Probability of Default)

## 1. Vue d'ensemble Architecture PD

### 1.1 Contexte

Plateforme de credit risk modeling pour un client fintech (transferts internationaux). Dataset proxy : Freddie Mac Single Family Loan-Level Dataset. Stack technique : XGBoost/LightGBM/Logistic Regression, MLflow, FastAPI, Docker, PostgreSQL.

### 1.2 Architecture Globale

L'architecture PD s'organise en trois couches distinctes :

- **Couche Pipeline** (`src/pipelines/Features/`) : construction et transformation des features
- **Couche Training** (`src/PDcomponent/run/`) : entraînement des modèles avec Optuna
- **Couche Serving** (`src/api/`) : API REST pour l'inférence en temps réel

### 1.3 Hiérarchie des Classes

```
RunAbstraction (ABC)
    ↓
PDRun (hérite de RunAbstraction)
    ↓
XGBoostRun, LightGBMRun, LogistiqueRegressionTrainRun

FeaturePipeline (ABC)
    ↓
PDFeaturePipeline (hérite de FeaturePipeline)

PredictionAbstraction (ABC)
    ↓
PDPrediction (hérite de PredictionAbstraction)
```

Cette hiérarchie permet une séparation claire des responsabilités : l'abstraction gère le contrat, les implémentations concrètes gèrent les spécificités algorithmiques.

---

## 2. Pipeline de Features

### 2.1 Responsabilités des Composants

#### 2.1.1 CleaningPipeline (`cleaningPipeline.py`)

**Responsabilité unique** : nettoyage et préparation des données brutes avant toute construction de features.

**Fonctionnalités clés** :
- Nettoyage de la table origination (`orig_impute`) : traitement des valeurs manquantes, imputation par médiane segmentée (VINTAGE × PROPERTY_TYPE), création d'indicateurs de missingness
- Nettoyage de la table historique (`hist_impute`) : conversion des dates, traitement des codes de délinquance, imputation de REMAINING_MONTHS_TO_LEGAL_MATURITY
- Jointure origination/historique avec enrichissement (ORIGINAL_LOAN_TERM, ORIGINAL_UPB, LTV)
- Sélection des colonnes éligibles (ELIGIBLE_COLUMNS pour hist, ORIGINATION_ELIGIBLE pour orig)

**Décision technique** : imputation par médiane segmentée plutôt que globale pour préserver les patterns temporels et sectoriels.

#### 2.1.2 WindowBuilder (`window_builder.py`)

**Responsabilité unique** : découpage du panel mensuel en fenêtres d'observation (12 mois par défaut).

**Fonctionnalité** : `groupby(LOAN_SEQUENCE_NUMBER).tail(12)` pour ne conserver que les 12 derniers mois par prêt.

**Décision technique** : bypassé à l'inférence — le Warehouse fournit déjà les données fenêtrées sur 12 mois, évitant un recalcul inutile.

#### 2.1.3 DelinquencyFeatures (`delinquency_features.py`)

**Responsabilité** : construction des features comportementales de retard de paiement.

**Features calculées** (vectorisé, un seul groupby) :
- `freq` : fréquence des mois en retard
- `severite` : DPD moyen
- `profondeur_max` : DPD maximum
- `n_profondeur_max` : nombre d'occurrences du DPD max
- `tendance` : slope de l'évolution du DPD
- `recuperation` : durée moyenne des épisodes de retard
- Combinaisons : `freq_x_profondeur_max`, `freq_x_tendance`, `freq_x_recuperation`, `recidivisme_extreme`

**Optimisation** : transforms partagés calculés une seule fois dans `__init__` pour éviter les recalculs.

#### 2.1.4 CapitalFeatures (`capital_features.py`)

**Responsabilité** : construction des features liées au capital et au remboursement.

**Features calculées** (4 angles) :
- **Niveau** : ratio UPB final / UPB original
- **Progression** : std des différences mensuelles d'UPB
- **Écart au plan** : différence entre UPB théorique (amortissement) et UPB réel
- **Anticipation** : proportion des remboursements supérieurs à la mensualité théorique

**Décision technique** : calcul de l'UPB théorique avec formule actuarielle (taux d'intérêt réel) pour détecter les remboursements anticipés.

#### 2.1.5 OriginationFeatures (`origination_features.py`)

**Responsabilité** : construction des features statiques d'origination selon les 5C du crédit.

**Features par dimension** :
- **C1 - Capacity** : mensualité implicite, charge taux × durée
- **C2 - Capital** : pression levier (LTV × DTI), écart LTV/OCLTV, couverture MI
- **C3 - Collateral** : multi-unité, occupancy risk
- **C4 - Conditions** : produit risque (IO/ARM), refi flag
- **C5 - Character** : segment de crédit, primo-accédant, co-emprunteur

**Décision technique** : vectorisation complète (pas d'apply) pour performance.

#### 2.1.6 FeatureSelector (`feature_selector.py`)

**Responsabilité** : sélection et mise à l'échelle des features basées sur les décisions EDA.

**Fonctionnalités** :
- Suppression de features non discriminantes (TO_DROP)
- Scaling différencié : RobustScaler pour features sensibles aux outliers, StandardScaler pour distributions normales, None pour catégorielles
- Logging MLflow des scalers avec métadonnées

**Décision technique** : 2 appels sklearn au lieu de 19 (un par groupe de scaling) pour optimisation.

#### 2.1.7 WoePipeline (`woe_pipeline.py`)

**Responsabilité** : discrétisation optimale et transformation WoE (Weight of Evidence) via optbinning.

**Fonctionnalités** :
- `BinningProcess` avec sélection par IV (Information Value) ≥ 0.02
- Transformation WoE des features sélectionnées
- Rapports : IV par feature, sélection/rejet, tables de binning

**Décision technique** : cache de summary() et selected() dans `__init__` pour éviter les recalculs.

#### 2.1.8 PDFeaturePipeline (`pdFeaturePipeline.py`)

**Responsabilité** : spécialisation du FeaturePipeline pour le modèle PD (target binaire, WoE, SMOTE).

**Fonctionnalités spécifiques** :
- Target binaire : défaut = 1 si max(DPD) ≥ 3 sur 12 mois (standard IFRS 9 / Bâle III)
- Transformation WoE via `apply_woe()` : fit sur train, réutilisation sur test/inference
- Rééquilibrage SMOTE sur train uniquement

**Flux d'exécution** :
```python
# Train
X, y = pipeline.build(hist, orig)  # features brutes scalées
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_woe, y_train_bal = pipeline.apply_woe(X_train, y_train)  # WoE + SMOTE
X_test_woe, _ = pipeline.apply_woe(X_test)  # WoE sans SMOTE

# Inference
pipeline_inf = PDFeaturePipeline(state="inference", binning_process=pipeline.binning_process)
X_new = pipeline_inf.build(hist_new, orig_new, scaler)
X_new_woe, _ = pipeline_inf.apply_woe(X_new)
```

---

## 3. Stratégie Train/Val/Test et Anti-Leakage

### 3.1 Split Temporel

Le split est effectué **avant** toute transformation WoE pour éviter le data leakage. Les maps train/val/test sont fournies en amont à `RunAbstraction`.

### 3.2 Points d'Anti-Leakage

#### 3.2.1 Split Externe avant WoE

**Risque** : si WoE est calculé sur l'ensemble du dataset avant split, les bins contiennent des informations du test set.

**Solution** : WoE est calculé uniquement sur train via `apply_woe(X_train, y_train)`. Le `binning_process` est ensuite réutilisé sur val/test via `apply_woe(X_val)` et `apply_woe(X_test)`.

**Code clé** (`pdFeaturePipeline.py`) :
```python
def apply_woe(self, X, y=None):
    if y is not None:
        self.woe_pipeline_ = WoePipeline(X, y, config=self.woe_config)
        self.binning_process = self.woe_pipeline_.capturedFit
        X_woe = self.woe_pipeline_.transform()
        X_woe, y = self._balance(X_woe, y)  # SMOTE
        return X_woe, y
    # Réutilisation du binning_process appris
    woe_inf = WoePipeline(X, config=self.woe_config, binning_process=self.binning_process)
    return woe_inf.transform(), None
```

#### 3.2.2 SMOTE sur Train Uniquement

**Risque** : appliquer SMOTE sur val/test introduit des données synthétiques dans l'évaluation.

**Solution** : SMOTE est appelé uniquement dans la branche `y is not None` de `apply_woe()`, donc uniquement sur train.

#### 3.2.3 Scaler Fitté sur Train

**Risque** : scaler fitté sur l'ensemble du dataset leak les statistiques de test.

**Solution** : `FeatureSelector.fit_transform()` est appelé uniquement sur train dans `FeaturePipeline.build()`. À l'inférence, `FeatureSelector.transform()` utilise le scaler chargé depuis MLflow.

**Code clé** (`featurePipeline.py`) :
```python
if self.state == "train":
    x, y = self.selector.fit_transform(data, target=self.target)
    self.scaler_run_id = self.selector.scaler_run_id
    return x, y
# inférence — pas de fit
selector = FeatureSelector()
return selector.transform(data, scaler)
```

#### 3.2.4 Calibration sur Validation Set

**Risque** : calibrer sur train surfit le modèle.

**Solution** : calibration avec `cv="prefit"` sur validation set uniquement.

**Code clé** (`xgboostrun.py`, `ligthGBMRUN.py`, `logistiqueRegressionTrainRun.py`) :
```python
calibrated_model = CalibratedClassifierCV(
    estimator=self._model_artifact,
    method=calibration_method,
    cv=calibration_CV  # "prefit"
)
calibrated_model.fit(x_val_transformed, y_val)
```

#### 3.2.5 Threshold Search sur Validation Set

**Risque** : optimiser le seuil sur test overfit l'évaluation.

**Solution** : seuil optimisé sur validation set via `threshold()`, puis appliqué tel quel sur test via `apply_threshold()`.

**Code clé** (`pdRUN.py`) :
```python
# Validation — recherche du seuil optimal
handeler = self.threshold(y_val, y_proba)
chosen_threshold = handeler['threshold']

# Test — application du seuil figé
recall, precision, f1, predicted_new, threshold = self.apply_threshold(y_test, y_prob)
```

### 3.3 Pipeline Dédié pour Validation

**Décision technique** : un pipeline dédié `pipeline_val` est instancié pour la validation, avec le même `binning_process` que train mais sans refit. Cela garantit que la transformation WoE est identique tout en évitant tout leakage.

**Code clé** (`xgboostrun.py`) :
```python
pipeline_val = PDFeaturePipeline(
    window_months=12,
    woe_config=self.config['woe'],
    binning_process=binning_process  # réutilisation du binning_process train
)
x_val_transformed, _ = pipeline_val.apply_woe(self._x_val)
```

---

## 4. Serving — Flow d'Inférence

### 4.1 Architecture API

```
GET /pd/predict/{loan_id}
    ↓
PDModelService.predict()
    ↓
WarehouseReader.fetch(loan_id)
    ↓
PDPrediction (hist, orig)
    ↓
PDFeaturePipeline.build() + apply_woe()
    ↓
Model.predict_proba()
    ↓
{"loan_id": "...", "pd_probability": 0.XX}
```

### 4.2 Composants Serving

#### 4.2.1 Route (`pd_model_route.py`)

**Responsabilité** : endpoint REST FastAPI.

```python
@router.get("/predict/{loan_id}")
def predict(loan_id: str) -> dict:
    proba = service.predict(loan_id)
    return {"loan_id": loan_id, "pd_probability": proba}
```

#### 4.2.2 Service (`pd_modelService.py`)

**Responsabilité** : orchestration de l'inférence.

**Flux** :
1. Charge hist/orig depuis Warehouse via `WarehouseReader.fetch(loan_id)`
2. Instancie `PDPrediction` avec les configs MLflow et modèle
3. Retourne la probabilité prédite

#### 4.2.3 Warehouse (`warehouseLoader.py`)

**Responsabilité** : accès aux données PostgreSQL via SQLAlchemy.

**Tables** :
- `loans_performance` : données historiques fenêtrées (12 mois)
- `loans_origination` : données statiques d'origination

**Décision technique** : les données sont déjà fenêtrées dans le Warehouse (12 mois), donc `WindowBuilder` est bypassé à l'inférence.

**Code clé** (`warehouseLoader.py`) :
```python
def fetch(self, loan_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_hist = text("SELECT * FROM loans_performance WHERE loan_sequence_number = :loan_id")
    query_orig = text("SELECT * FROM loans_origination WHERE loan_sequence_number = :loan_id")
    # ... lecture SQL
    hist.columns = hist.columns.str.upper()  # normalisation
    orig.columns = orig.columns.str.upper()
    return hist, orig
```

#### 4.2.4 PDPrediction (`PDprediction.py`)

**Responsabilité** : inférence PD via `PredictionAbstraction`.

**Flux** :
1. Charge les artefacts MLflow (binning_process, model_fit, scaler)
2. Instancie `PDFeaturePipeline` avec `state="prediction"` et le `binning_process` chargé
3. Applique le pipeline : `build()` (scaling) → `apply_woe()` (transformation WoE)
4. Prédit via `model_fit.predict_proba()`

**Code clé** (`PDprediction.py`) :
```python
def setup(self):
    binning_process, self._model_fit = self._load_data()
    scaler = self._load_scaler()
    self._featurePipeline = PDFeaturePipeline(
        window_months=12,
        woe_config=self._model_config['woe'],
        binning_process=binning_process,
        state='prediction'  # important : pas de fit
    )
    self._x_data = self._featurePipeline.build(self._hist, self._orig, scaler)

def apply(self):
    x_woe, _ = self._featurePipeline.apply_woe(self._x_data)
    return self._model_fit.predict_proba(x_woe)[:, 1]
```

### 4.3 Bypass de WindowBuilder

À l'inférence, `WindowBuilder` est bypassé car le Warehouse fournit déjà les données fenêtrées sur 12 mois. Cela est géré dans `FeaturePipeline._build_features()` :

```python
if self.state == "train":
    hist_12m = WindowBuilder(hist, window_months=self.window_months).build()
else:
    hist_12m = hist  # déjà fenêtré
```

---

## 5. Décisions Techniques et Justifications

### 5.1 Architecture

| Décision | Justification |
|----------|--------------|
| ABC pour pipelines (FeaturePipeline, RunAbstraction, PredictionAbstraction) | Réutilisabilité pour LGD/EAD, séparation claire des contrats |
| Pipeline dédié pour validation | Évite le leakage tout en garantissant une transformation identique |
| WindowBuilder bypassé à l'inférence | Optimisation — données déjà fenêtrées dans Warehouse |

### 5.2 Feature Engineering

| Décision | Justification |
|----------|--------------|
| Imputation médiane segmentée (VINTAGE × PROPERTY_TYPE) | Préserve les patterns temporels et sectoriels |
| Vectorisation complète (pas d'apply) | Performance sur dataset Freddie Mac volumineux |
| 4 angles pour capital (niveau, progression, écart, anticipation) | Capture multi-dimensionnelle du comportement de remboursement |
| 5C du crédit pour origination | Standard industriel, interprétabilité métier |
| RobustScaler pour features sensibles aux outliers | Robustesse aux valeurs extrêmes (DTI, LTV) |
| StandardScaler pour distributions normales | Meilleure performance pour modèles linéaires |

### 5.3 Modélisation

| Décision | Justification |
|----------|--------------|
| Target binaire DPD ≥ 3 sur 12 mois | Standard IFRS 9 / Bâle III |
| WoE avec optbinning (IV ≥ 0.02) | Discrétisation optimale, réduction de dimension, interprétabilité |
| SMOTE sur train uniquement | Évite le data leakage, rééquilibre les classes |
| Calibration cv="prefit" sur validation | Fiabilité des probabilités sans overfit |
| Threshold search par percentiles | Pratique crédit (percentiles plutôt que grille fixe) |
| Contrainte recall ≥ 0.90 éliminatoire | Priorité métier : ne pas manquer les défauts |
| Optuna pour hyperparamètres | Efficacité, parallélisation, early stopping |

### 5.4 MLOps

| Décision | Justification |
|----------|--------------|
| MLflow pour tracking | Standard industriel, reproductibilité |
| Scaler loggé séparément (run dédié) | Réutilisabilité cross-modèles |
| Artefacts versionnés (binning_process, model_fit, scaler) | Reproductibilité totale de l'inférence |
| Config externe (YAML) | Flexibilité, séparation code/config |
| Seuil figé dans config test après validation | Déploiement déterministe |

---

## 6. Points Ouverts

### 6.1 Évaluation

Les métriques finales d'évaluation ne sont pas encore documentées :
- ROC-AUC et Gini sur test set
- Matrice de confusion (TN, FP, FN, TP)
- Recall, Precision, F1 au seuil optimal
- Courbes ROC, Precision-Recall, Calibration

Ces métriques sont loggées dans MLflow mais pas encore consolidées dans un rapport métier.

### 6.2 Comparaison des Modèles

La comparaison entre XGBoost, LightGBM et Logistic Regression n'est pas finalisée :
- Performance discriminante (ROC-AUC, Gini)
- Calibration des probabilités (Brier score, calibration curve)
- Interprétabilité (SHAP values pour tree-based, coefficients pour LR)
- Temps d'entraînement et d'inférence
- Robustesse aux drifts de données

### 6.3 Conclusions Métier

L'impact métier des résultats n'est pas encore formalisé :
- Segmentation des prêts par risque (low/medium/high)
- Taux d'acceptation vs taux de défaut attendu
- Impact sur le capital réglementaire (Bâle III)
- Recommandations de politique de crédit

### 6.4 Améliorations Possibles

- **Feature selection automatique** : actuellement basée sur EDA manuel, pourrait être automatisée (RFE, SHAP)
- **Cross-validation temporelle** : pour tester la robustesse aux drifts temporels
- **Monitoring en production** : drift detection, alertes sur dégradation de performance
- **A/B testing** : pour comparer les modèles en production
- **Explainability** : SHAP values intégrées dans l'API pour transparence métier

---

## 7. Annexes

### 7.1 Extrait de Code — Anti-Leakage WoE

```python
# pdFeaturePipeline.py
def apply_woe(self, X, y=None):
    if y is not None:
        # Train : fit WoE + SMOTE
        self.woe_pipeline_ = WoePipeline(X, y, config=self.woe_config)
        self.binning_process = self.woe_pipeline_.capturedFit
        X_woe = self.woe_pipeline_.transform()
        X_woe, y = self._balance(X_woe, y)  # SMOTE
        return X_woe, y
    
    # Test/Inference : réutilisation du binning_process
    if self.binning_process is None:
        raise RuntimeError("binning_process manquant")
    woe_inf = WoePipeline(X, config=self.woe_config, binning_process=self.binning_process)
    return woe_inf.transform(), None
```

### 7.2 Extrait de Code — Threshold Optimization

```python
# pdRUN.py
def threshold(self, y_data, y_proba):
    percentiles = np.arange(1, 100, 1)  # 1er à 99e percentile
    thresholds = np.percentile(y_proba, percentiles)
    
    constraints = self.config['evaluation']['threshold']['constraints']
    recall_min = constraints['recall_min']  # ex: 0.90
    f1_min = constraints['f1_min']  # ex: 0.70
    
    best_f1 = -1
    chosen_threshold = 0
    
    for pct, t in zip(percentiles, thresholds):
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_data, y_pred, zero_division=0)
        f1 = f1_score(y_data, y_pred, zero_division=0)
        
        if r >= recall_min and f1 > best_f1:
            best_f1 = f1
            chosen_threshold = t
    
    return {'threshold': chosen_threshold, 'recall': r, 'f1': f1, ...}
```

### 7.3 Extrait de Code — Inference Flow

```python
# pd_modelService.py
def predict(self, loan_id: str) -> float:
    hist, orig = self._reader.fetch(loan_id)
    prediction = PDPrediction(
        hist=hist,
        orig=orig,
        mlflow_config=os.environ["MLFLOW_CONFIG_PATH"],
        model_config=os.environ["MODEL_CONFIG_PATH"]
    )
    return float(prediction.apply()[0])

# PDprediction.py
def setup(self):
    binning_process, self._model_fit = self._load_data()
    scaler = self._load_scaler()
    self._featurePipeline = PDFeaturePipeline(
        window_months=12,
        woe_config=self._model_config['woe'],
        binning_process=binning_process,
        state='prediction'
    )
    self._x_data = self._featurePipeline.build(self._hist, self._orig, scaler)

def apply(self):
    x_woe, _ = self._featurePipeline.apply_woe(self._x_data)
    return self._model_fit.predict_proba(x_woe)[:, 1]
```

---

**Document version** : 1.0  
**Date** : 26 juin 2026  
**Auteur** : Audit technique pipeline PD  
**Scope** : Architecture PD, pipeline features, anti-leakage, serving
