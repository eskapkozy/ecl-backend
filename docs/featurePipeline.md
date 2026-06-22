# Feature Pipeline — Architecture & Réutilisabilité
**Projet** : Outil de Recommandation de Crédit — Gamme ECL (PD / LGD / EAD)
**Sujet** : Stratégie de réutilisation du pipeline de features entre modèles
**Date** : Juin 2026

---

## Constat de départ

Le pipeline de feature engineering construit pour le modèle PD repose sur des modules génériques (fenêtre d'observation, comportement de retard, comportement de capital, caractéristiques d'origination) qui décrivent le **profil et le comportement du prêt** — une information indépendante du modèle qui la consomme ensuite.

Les modèles LGD (perte en cas de défaut) et EAD (exposition au moment du défaut) à venir utiliseront le même historique de prêts Freddie Mac et bénéficient donc potentiellement des mêmes features. Ce document fixe la stratégie à suivre pour éviter de dupliquer le code tout en respectant le principe de responsabilité unique.

---

## Ce qui est réutilisable tel quel

| Module | Réutilisable pour LGD/EAD | Raison |
|---|---|---|
| `window_builder.py` | Oui | La logique de fenêtre d'observation est indépendante de la target |
| `delinquency_features.py` | Oui | Le comportement de retard est un signal pertinent pour PD, LGD et EAD |
| `capital_features.py` | Oui | L'exposition et la trajectoire du capital sont pertinentes pour les trois modèles |
| `origination_features.py` | Oui | Les caractéristiques statiques (5C) ne dépendent pas du modèle cible |

---

## Ce qui est spécifique au PD et ne doit pas être réutilisé directement

| Élément | Pourquoi c'est spécifique au PD |
|---|---|
| `_build_target()` | Target binaire (défaut/non-défaut). LGD et EAD ont des targets continues (% de perte, montant d'exposition) |
| `apply_woe()` | Le WoE est une technique de scoring pour variable cible binaire — non pertinente pour une régression |
| `_balance()` (SMOTE) | Le rééquilibrage de classes n'a de sens que pour une classification |
| `self.task = "CLASSIFICATION"` | Doit devenir `"REGRESSION"` pour LGD et EAD |

---

## Architecture cible — héritage par responsabilité

Le principe retenu est l'**héritage avec une classe de base commune**, suivant le principe ouvert/fermé : le code commun et coûteux (construction des features) reste stable, seules les spécificités de chaque modèle s'ajoutent par sous-classe.

```
FeaturePipeline (classe de base abstraite)
    _build_features()      →  commun à tous les modèles — fenêtre, retard, capital, origination
    _impute()               →  commun
    build()                 →  orchestration commune, appelle _build_target() (abstraite)

PDFeaturePipeline(FeaturePipeline)
    _build_target()         →  binaire — défaut = 1 si max(DPD) >= 3 sur 12 mois
    apply_woe()              →  WoE + SMOTE — spécifique scoring binaire

LGDFeaturePipeline(FeaturePipeline)
    _build_target()         →  continue — % de perte sur les prêts en défaut
    apply_scaling()          →  scaler seul, pas de WoE ni SMOTE

EADFeaturePipeline(FeaturePipeline)
    _build_target()         →  continue — montant d'exposition au moment du défaut
    apply_scaling()          →  scaler seul, pas de WoE ni SMOTE
```

---

## Séquence d'utilisation — pattern commun aux trois modèles

```
1. build(hist, orig)                    →  features brutes scalées (X, y) — AVANT split
2. train_test_split(X, y)                →  fait par l'utilisateur, hors pipeline
3. apply_woe() ou apply_scaling()        →  transformation finale — APRÈS split, train uniquement
4. apply_woe() / apply_scaling() (test)  →  réutilise l'artefact appris en train
```

Cette séquence évite toute fuite d'information : le WoE (ou le scaler pour LGD/EAD) et le rééquilibrage de classes ne sont jamais appris sur les données de test.

---

## Ce qu'il faudra faire au moment d'attaquer LGD et EAD

1. Créer `PDFeaturePipeline`, `LGDFeaturePipeline`, `EADFeaturePipeline` héritant de `FeaturePipeline`
2. Déplacer `_build_target()` (logique PD actuelle) dans `PDFeaturePipeline`
3. Déplacer `apply_woe()` et `_balance()` dans `PDFeaturePipeline`
4. Construire `_build_target()` pour LGD — vérifier la disponibilité des variables de perte/recouvrement dans le dataset Freddie Mac
5. Construire `_build_target()` pour EAD — vérifier la disponibilité des variables d'exposition au défaut
6. Créer `apply_scaling()` pour LGD/EAD — RobustScaler ou StandardScaler simple, sans WoE ni SMOTE
7. Revalider `feature_selector.py` — le `SCALING` dict actuel est calibré sur l'EDA du PD ; LGD et EAD nécessiteront leur propre EDA et donc leurs propres décisions de sélection

---

## Référence

Ce document doit être consulté avant toute reprise du projet sur les modèles LGD ou EAD, pour appliquer la même discipline d'architecture que celle établie sur le PD.