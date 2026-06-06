# Dictionnaire des Variables — Freddie Mac Single Family Loan-Level Dataset (FSSL)
## Fichier : Origination

> **Source** : Freddie Mac Single Family Loan-Level Dataset  
> **Format** : Pipe-delimited (`|`), sans en-tête  
> **Grain** : Une ligne = un prêt à l'origination  
> **Dernière mise à jour du dictionnaire** : 2025

---

## Vue d'ensemble

Le fichier d'origination contient les caractéristiques **statiques** du prêt et de l'emprunteur telles qu'elles sont connues au moment de l'octroi du crédit. Ces données constituent le socle de tout modèle de **scoring à froid (Application Scorecard)** ainsi que la base de référence pour les modèles comportementaux.

---

## Dictionnaire des Variables

| # | Nom de Variable | Type | Valeurs / Format | Description | Réglementation / Dépendances |
|---|----------------|------|-----------------|-------------|------------------------------|
| 1 | `CREDIT_SCORE` | Numérique | 300 – 850 / `9999` si manquant | Score de crédit FICO de l'emprunteur principal à l'origination. Reflète le risque de crédit historique. | Bâle II/III (IRB) : facteur clé dans l'estimation de la PD. Corrélé à `DTI` et `LTV`. Freddie Mac exige généralement un score ≥ 620. |
| 2 | `FIRST_PAYMENT_DATE` | Date | `YYYYMM` | Date du premier remboursement du prêt. Permet de situer le prêt dans le cycle économique. | Sert de référence pour calculer l'ancienneté du prêt (`loan age`) dans le fichier performance. Dépend de `MATURITY_DATE` et `ORIGINAL_LOAN_TERM`. |
| 3 | `FIRST_TIME_HOMEBUYER_FLAG` | Catégoriel | `Y` = Oui / `N` = Non / `9` = Inconnu | Indique si l'emprunteur est un primo-accédant à la propriété. | Programmes publics d'aide (FHFA, HUD) : impacte les conditions d'éligibilité et le niveau de risque attendu. Corrélé à `LOAN_PURPOSE`. |
| 4 | `MATURITY_DATE` | Date | `YYYYMM` | Date d'échéance finale du prêt (dernier remboursement prévu). | Déduite de `FIRST_PAYMENT_DATE` + `ORIGINAL_LOAN_TERM`. Toute incohérence entre ces trois variables est une anomalie structurelle. |
| 5 | `MSA` | Catégoriel | Code numérique 5 chiffres / vide si zone rurale | Code de la zone statistique métropolitaine (Metropolitan Statistical Area) du bien financé. | Défini par le U.S. Office of Management and Budget (OMB). Vide = zone non métropolitaine. Utilisé pour les analyses géographiques du risque. |
| 6 | `MI_PERCENTAGE` | Numérique | `000` – `055` (en %) / `000` si aucune assurance | Pourcentage de couverture de l'assurance hypothécaire privée (PMI) souscrite sur le prêt. | Réglementation Freddie Mac : obligatoire si `LTV` > 80%. Impacte directement le calcul de la **LGD**. Dépendance forte avec `LTV` et `OCLTV`. |
| 7 | `NUMBER_OF_UNITS` | Numérique | `1`, `2`, `3`, `4` | Nombre de logements dans le bien immobilier financé. | Freddie Mac limite la couverture aux biens de 1 à 4 unités. Influe sur le profil de risque et les plafonds de prêt conformes (conforming loan limits). |
| 8 | `OCCUPANCY_STATUS` | Catégoriel | `P` = Résidence principale / `S` = Résidence secondaire / `I` = Investissement | Statut d'occupation déclaré par l'emprunteur. | Variable de risque clé : les prêts investissement (`I`) ont historiquement des taux de défaut plus élevés. Impacte la PD et la LGD. Corrélé à `LOAN_PURPOSE`. |
| 9 | `OCLTV` | Numérique | 1 – 200 (en %) / `999` si manquant | Combined Loan-to-Value à l'origination : ratio de l'ensemble des dettes garanties par le bien sur la valeur du bien. | Bâle III : facteur central dans le calcul des RWA (Risk-Weighted Assets). Différent de `LTV` car inclut les prêts subordonnés. Si `OCLTV` > `LTV` → présence probable d'un second prêt. |
| 10 | `DTI` | Numérique | 1 – 65 (en %) / `999` si manquant | Ratio dette sur revenu (Debt-to-Income) : part des charges de dette mensuelles dans le revenu brut mensuel de l'emprunteur. | Freddie Mac : DTI ≤ 45% en règle générale (exceptions jusqu'à 50% avec compensateurs). Indicateur de capacité de remboursement. Corrélé à `CREDIT_SCORE` et `UPB`. |
| 11 | `UPB` | Numérique | Montant en USD | Solde du capital impayé (Unpaid Principal Balance) à l'origination. Montant initial du prêt. | Base de calcul de l'`EAD` (Exposure at Default). Soumis aux conforming loan limits fixés annuellement par la FHFA. Corrélé à `LTV` et à la valeur du bien. |
| 12 | `LTV` | Numérique | 1 – 105 (en %) / `999` si manquant | Loan-to-Value à l'origination : ratio du montant du prêt sur la valeur d'expertise du bien. | Bâle II/III : variable déterminante pour la LGD. Freddie Mac : LTV > 80% déclenche l'obligation d'assurance hypothécaire (`MI_PERCENTAGE` > 0). Dépendance directe avec `UPB` et valeur du bien. |
| 13 | `ORIGINAL_INTEREST_RATE` | Numérique | Format décimal (ex : `7` = 7.000%) | Taux d'intérêt nominal annuel à l'origination du prêt. | Pour les prêts à taux fixe (`FRM`), ce taux est constant. Pour les ARM, ce taux est le taux initial. Influe sur la charge mensuelle et donc sur le `DTI`. |
| 14 | `CHANNEL` | Catégoriel | `R` = Retail / `B` = Broker / `C` = Correspondent / `T` = TPO | Canal de distribution par lequel le prêt a été accordé. | Variable de risque opérationnel : les prêts broker (`B`) et correspondent (`C`) présentent historiquement des profils de risque différents des prêts retail. |
| 15 | `PPM_FLAG` | Catégoriel | `Y` = Oui / `N` = Non | Indique si le prêt est assorti d'une pénalité de remboursement anticipé (Prepayment Penalty Mortgage). | Réglementation Dodd-Frank / CFPB : encadrement strict des pénalités de prépaiement. Impacte la modélisation du risque de prépaiement (CPR). |
| 16 | `PRODUCT_TYPE` | Catégoriel | `FRM` = Taux fixe / `ARM` = Taux variable | Type de produit hypothécaire. | Variable structurante : détermine la nature de l'évolution des flux de remboursement. Les ARM impliquent un risque de taux additionnel. Dépendance avec `ORIGINAL_INTEREST_RATE`. |
| 17 | `STATE` | Catégoriel | Code État US à 2 lettres (ex : `IL` = Illinois) | État fédéral où est situé le bien financé. | Les procédures de saisie (foreclosure) varient fortement selon l'État (judiciaire vs non-judiciaire), ce qui impacte directement la **LGD** et le temps de recouvrement. Variable géographique de risque. |
| 18 | `PROPERTY_TYPE` | Catégoriel | `SF` = Single Family / `CO` = Condo / `PU` = PUD / `MH` = Manufactured Housing / `CP` = Co-op | Type de bien immobilier financé. | Influe sur la liquidité du bien en cas de saisie et donc sur la LGD. Les manufactured housing (`MH`) présentent des LGD historiquement plus élevées. |
| 19 | `POSTAL_CODE` | Catégoriel | Code postal à 5 chiffres (3 premiers significatifs) | Code postal du bien. Freddie Mac ne publie que les 3 premiers chiffres pour des raisons de confidentialité. | Utilisé pour les analyses de risque géographique et la corrélation avec des données macroéconomiques locales (chômage, prix immobiliers). Lié à `MSA` et `STATE`. |
| 20 | `LOAN_SEQUENCE_NUMBER` | Alphanumérique | Format : `FYYQNNNNNNNNN` | Identifiant unique du prêt dans le dataset Freddie Mac. Clé de jointure entre le fichier origination et le fichier performance. | **Clé primaire.** Indispensable pour toute jointure entre les deux fichiers. Format : F + année 2 chiffres + trimestre + séquence. |
| 21 | `LOAN_PURPOSE` | Catégoriel | `P` = Purchase / `C` = Cash-out Refi / `N` = No cash-out Refi / `U` = Inconnu | Objet du prêt : acquisition, refinancement avec extraction de cash, ou refinancement sans extraction. | Variable de risque : les cash-out refis (`C`) augmentent le LTV et le niveau d'endettement. Corrélé à `FIRST_TIME_HOMEBUYER_FLAG` et `OCCUPANCY_STATUS`. |
| 22 | `ORIGINAL_LOAN_TERM` | Numérique | En mois (ex : `360` = 30 ans) | Durée initiale prévue du prêt en mois. | Doit être cohérent avec `FIRST_PAYMENT_DATE` et `MATURITY_DATE`. Les termes standards sont 180 (15 ans) et 360 (30 ans). Influe sur l'amortissement et le profil de risque. |
| 23 | `NUMBER_OF_BORROWERS` | Numérique | `1` à `10` / `99` si inconnu | Nombre total d'emprunteurs sur le prêt (emprunteur principal + co-emprunteurs). | La présence d'un co-emprunteur peut améliorer le profil de risque (revenu combiné plus élevé). Impacte l'interprétation du `DTI` et du `CREDIT_SCORE`. |
| 24 | `SELLER_NAME` | Catégoriel | Nom de l'établissement / `Other sellers` | Nom de l'institution financière qui a vendu le prêt à Freddie Mac. | Permet l'analyse de la qualité de souscription par vendeur. Utilisé dans les audits de qualité et les analyses de back-testing. |
| 25 | `SERVICER_NAME` | Catégoriel | Nom de l'établissement / `Other servicers` | Nom de l'entreprise en charge de la gestion administrative du prêt (collecte des paiements, gestion des défauts). | Le servicer influence les taux de modification de prêt et les délais de résolution en cas de défaut, donc la LGD. |
| 26 | `SUPER_CONFORMING_FLAG` | Catégoriel | `Y` = Oui / vide = Non | Indique si le prêt dépasse les limites conformes standard mais reste dans les limites super-conformes (zones à coût élevé). | Limites fixées annuellement par la FHFA. Prêts autorisés dans les zones géographiques à prix immobiliers élevés. Lié à `UPB`, `STATE` et `MSA`. |
| 27 | `PRE_RELIEF_REFI_LOAN_SEQ` | Alphanumérique | Numéro de séquence / vide | Identifiant du prêt original avant refinancement dans le cadre du programme HARP (Home Affordable Refinance Program). | Spécifique aux prêts refinancés sous HARP. Permet de tracer l'historique du prêt. Lié à `LOAN_PURPOSE`. |
| 28 | `PROGRAM_INDICATOR` | Catégoriel | `H` = Home Possible / `9` = Autre / vide | Indicateur du programme d'aide ou d'accessibilité sous lequel le prêt a été accordé. | Home Possible est un programme Freddie Mac pour emprunteurs à revenus modérés. Influe sur les critères d'éligibilité et le profil de risque attendu. |
| 29 | `RELIEF_REFINANCE_INDICATOR` | Catégoriel | `Y` = Oui / vide = Non | Indique si le prêt est issu d'un refinancement de secours (relief refinance), notamment HARP. | Programme réglementaire fédéral post-crise 2008. Ces prêts peuvent avoir des LTV > 100% (prêts sous-marins). Lié à `LOAN_PURPOSE` et `LTV`. |
| 30 | `PROPERTY_VALUATION_METHOD` | Catégoriel | `1` = Full appraisal / `2` = Other / `3` = AVM / `9` = Inconnu | Méthode utilisée pour évaluer la valeur du bien à l'origination. | Impacte la fiabilité du `LTV` calculé. Les AVM (Automated Valuation Models) sont moins robustes que les expertises complètes, surtout en période de stress. Réglementation Dodd-Frank sur les appraisals. |
| 31 | `IO_FLAG` | Catégoriel | `Y` = Interest Only / `N` = Non | Indique si le prêt est de type Interest Only (remboursement des intérêts uniquement pendant une période initiale). | Prêts à risque plus élevé : le capital n'est pas amorti pendant la période IO, ce qui maintient un LTV élevé. Impacte fortement la modélisation de l'amortissement et la LGD. |
| 32 | `MORTGAGE_INSURANCE_CANCELLATION_INDICATOR` | Catégoriel | `Y` = Annulée / `N` = Non / `7` = N/A | Indique si l'assurance hypothécaire a été annulée avant la fin du prêt. | Loi Homeowners Protection Act (HPA) de 1998 : l'emprunteur peut demander l'annulation du PMI lorsque le LTV atteint 80%. Dépendance directe avec `LTV`, `MI_PERCENTAGE` et l'évolution de la valeur du bien. |

---

## Dépendances Réglementaires et Structurelles Clés

```
LTV ──────────────────────────────────── MI_PERCENTAGE (LTV > 80% → PMI obligatoire)
LTV + OCLTV ─────────────────────────── Détection de second prêt (OCLTV > LTV)
FIRST_PAYMENT_DATE + ORIGINAL_LOAN_TERM ─ MATURITY_DATE (cohérence obligatoire)
LOAN_SEQUENCE_NUMBER ────────────────── Clé de jointure → Fichier Performance
UPB + LTV ───────────────────────────── Valeur implicite du bien
DTI + UPB + ORIGINAL_INTEREST_RATE ──── Capacité de remboursement
STATE ────────────────────────────────── Procédure de saisie → Impact LGD
LTV + IO_FLAG ───────────────────────── Amortissement → Impact LGD
```

---

## Valeurs Manquantes Conventionnelles

| Code | Signification |
|------|--------------|
| `9999` | Valeur manquante pour `CREDIT_SCORE` |
| `999` | Valeur manquante pour `DTI`, `LTV`, `OCLTV` |
| `99` | Valeur manquante pour `NUMBER_OF_BORROWERS` |
| `9` | Valeur inconnue pour variables catégorielles |
| ` ` (vide) | Non applicable ou non renseigné |

---

## Notes d'Usage pour la Modélisation

> **Application Scorecard (scoring à froid)** : toutes les variables ci-dessus sont disponibles. Variables les plus discriminantes attendues : `CREDIT_SCORE`, `LTV`, `DTI`, `OCCUPANCY_STATUS`, `LOAN_PURPOSE`.

> **Behavioral Scorecard** : ces variables servent de base statique, enrichies par des variables agrégées issues du fichier performance via `LOAN_SEQUENCE_NUMBER`.

> **Limites structurelles** : absence de données de revenus absolus (seul le ratio `DTI` est disponible), pas de valeur exacte du bien (déductible de `UPB`/`LTV`), code postal tronqué aux 3 premiers chiffres.

---

## Eligibilité des Variables pour la PD

Cette section ne définit pas encore les variables candidates finales du modèle. Elle
identifie seulement les variables **utilisables sans leakage** selon le moment de
prédiction.

### Règle simple

| Cas | Décision |
|---|---|
| Variable connue à l'origination | Eligible pour `Origination PD` |
| Variable connue à l'origination et stable dans le temps | Eligible comme base statique pour `Behavioral PD` |
| Identifiant technique | Utilisable pour jointure, exclu comme feature |
| Variable trop granulaire | Eligible mais à encoder / regrouper prudemment |

### Variables éligibles sans leakage

| Groupe | Variables | Usage |
|---|---|---|
| Profil crédit | `CREDIT_SCORE`, `FIRST_TIME_HOMEBUYER_FLAG`, `NUMBER_OF_BORROWERS` | PD origination et behavioral |
| Capacité / levier | `DTI`, `LTV`, `OCLTV`, `MI_PERCENTAGE` | PD origination et behavioral |
| Montant / durée | `UPB`, `ORIGINAL_LOAN_TERM`, `MATURITY_DATE`, `FIRST_PAYMENT_DATE` | PD origination et behavioral |
| Produit | `ORIGINAL_INTEREST_RATE`, `PRODUCT_TYPE`, `IO_FLAG`, `PPM_FLAG` | PD origination et behavioral |
| Bien immobilier | `OCCUPANCY_STATUS`, `PROPERTY_TYPE`, `NUMBER_OF_UNITS`, `PROPERTY_VALUATION_METHOD` | PD origination et behavioral |
| Géographie | `STATE`, `MSA`, `POSTAL_CODE` | Eligible, encodage prudent |
| Origination | `CHANNEL`, `LOAN_PURPOSE`, `PROGRAM_INDICATOR`, `SUPER_CONFORMING_FLAG`, `RELIEF_REFINANCE_INDICATOR` | Eligible, à contrôler selon couverture |
| Acteurs | `SELLER_NAME`, `SERVICER_NAME` | Eligible, regrouper les modalités rares |

### Variables techniques ou à exclure comme features directes

| Variable | Décision | Raison |
|---|---|---|
| `LOAN_SEQUENCE_NUMBER` | Clé de jointure uniquement | Identifiant unique, pas une information de risque |
| `PRE_RELIEF_REFI_LOAN_SEQ` | Exclure au MVP | Identifiant d'un prêt antérieur, risque de surapprentissage |
| `POSTAL_CODE` | Eligible mais encodage prudent | Haute cardinalité et risque de surapprentissage géographique |
| `SELLER_NAME`, `SERVICER_NAME` | Eligible mais regroupement requis | Haute cardinalité, modalités rares |
| `MORTGAGE_INSURANCE_CANCELLATION_INDICATOR` | A contrôler avant usage | Peut représenter une information postérieure à l'origination selon le millésime |

### Décision pour le MVP

Pour le modèle `Origination PD`, on utilise uniquement les variables connues à
l'origination. Pour le modèle `Behavioral PD`, ces mêmes variables sont jointes à
chaque observation mensuelle via `LOAN_SEQUENCE_NUMBER`.

Les variables éligibles ne sont pas automatiquement retenues dans le modèle final :
elles devront ensuite passer les contrôles de qualité, couverture, stabilité,
cardinalité et importance métier.

---

*Dictionnaire établi sur la base du Freddie Mac Single Family Loan-Level Dataset User Guide.*
