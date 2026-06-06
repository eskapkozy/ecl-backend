# Dictionnaire des Variables - Freddie Mac Performance Dataset
## Fichier : Monthly Performance / SVCG

> **Source** : Freddie Mac Single Family Loan-Level Dataset  
> **Format** : Pipe-delimited (`|`), sans en-tête  
> **Grain** : Une ligne = un prêt observé sur un mois  
> **Clé de jointure** : `LOAN_SEQUENCE_NUMBER` avec le fichier Origination

---

## Vue d'ensemble

Le fichier Performance décrit l'évolution mensuelle d'un prêt après son origination :
encours courant, retard de paiement, âge du prêt, modification, assistance, sortie du
portefeuille, pertes et recouvrements.

Pour la modélisation PD, ce fichier doit être utilisé avec une règle temporelle stricte :

> A la date d'observation `t`, seules les informations connues jusqu'au mois `t` sont utilisables comme variables explicatives.

Les informations de sortie finale, de perte, de recouvrement ou de liquidation ne sont
pas des variables explicatives PD. Elles peuvent servir à construire des cibles ou à
alimenter les modèles LGD/EAD.

---

## Header attendu

```text
LOAN_SEQUENCE_NUMBER|MONTHLY_REPORTING_PERIOD|CURRENT_ACTUAL_UPB|CURRENT_LOAN_DELINQUENCY_STATUS|LOAN_AGE|REMAINING_MONTHS_TO_LEGAL_MATURITY|DEFECT_SETTLEMENT_DATE|MODIFICATION_FLAG|ZERO_BALANCE_CODE|ZERO_BALANCE_EFFECTIVE_DATE|CURRENT_INTEREST_RATE|CURRENT_NON_INTEREST_BEARING_UPB|DUE_DATE_OF_LAST_PAID_INSTALLMENT|MI_RECOVERIES|NET_SALE_PROCEEDS|NON_MI_RECOVERIES|TOTAL_EXPENSES|LEGAL_COSTS|MAINTENANCE_AND_PRESERVATION_COSTS|TAXES_AND_INSURANCE|MISCELLANEOUS_EXPENSES|ACTUAL_LOSS_CALCULATION|CUMULATIVE_MODIFICATION_COST|INTEREST_RATE_STEP_INDICATOR|PAYMENT_DEFERRAL_FLAG|ESTIMATED_LTV|ZERO_BALANCE_REMOVAL_UPB|DELINQUENT_ACCRUED_INTEREST|DELINQUENCY_DUE_TO_DISASTER|BORROWER_ASSISTANCE_STATUS_CODE|CURRENT_MONTH_MODIFICATION_COST|INTEREST_BEARING_UPB
```

---

## Dictionnaire des Variables

| # | Variable | Type | Description simple | Usage PD |
|---:|---|---|---|---|
| 1 | `LOAN_SEQUENCE_NUMBER` | Texte | Identifiant unique du prêt. | Jointure uniquement |
| 2 | `MONTHLY_REPORTING_PERIOD` | Date `YYYYMM` | Mois d'observation. | Eligible, axe temporel |
| 3 | `CURRENT_ACTUAL_UPB` | Numérique | Capital restant dû courant. | Eligible à `t` |
| 4 | `CURRENT_LOAN_DELINQUENCY_STATUS` | Catégoriel | Niveau de retard courant : `0`, `1`, `2`, `3+`, `RA`. | Eligible à `t` et cible future |
| 5 | `LOAN_AGE` | Numérique | Nombre de mois écoulés depuis le premier paiement. | Eligible à `t` |
| 6 | `REMAINING_MONTHS_TO_LEGAL_MATURITY` | Numérique | Mois restants jusqu'à la maturité contractuelle. | Eligible à `t` |
| 7 | `DEFECT_SETTLEMENT_DATE` | Date `YYYYMM` | Date de règlement d'un défaut de qualité / servicing. | Exclure PD MVP |
| 8 | `MODIFICATION_FLAG` | Catégoriel | Indique si le prêt a été modifié. | Eligible si connu à `t` |
| 9 | `ZERO_BALANCE_CODE` | Catégoriel | Raison de sortie du prêt du portefeuille. | Cible / exclusion feature |
| 10 | `ZERO_BALANCE_EFFECTIVE_DATE` | Date `YYYYMM` | Date effective de sortie du portefeuille. | Exclure feature PD |
| 11 | `CURRENT_INTEREST_RATE` | Numérique | Taux courant appliqué au prêt. | Eligible à `t` |
| 12 | `CURRENT_NON_INTEREST_BEARING_UPB` | Numérique | Partie du solde courant ne portant pas intérêt. | Eligible à `t` |
| 13 | `DUE_DATE_OF_LAST_PAID_INSTALLMENT` | Date `YYYYMM` | Dernière échéance effectivement payée. | Eligible à `t`, à contrôler |
| 14 | `MI_RECOVERIES` | Numérique | Recouvrements via assurance mortgage insurance. | LGD, exclure PD |
| 15 | `NET_SALE_PROCEEDS` | Numérique / `U` | Produit net de vente après liquidation. | LGD, exclure PD |
| 16 | `NON_MI_RECOVERIES` | Numérique | Recouvrements hors assurance. | LGD, exclure PD |
| 17 | `TOTAL_EXPENSES` | Numérique | Total des frais liés à liquidation / recouvrement. | LGD, exclure PD |
| 18 | `LEGAL_COSTS` | Numérique | Frais juridiques. | LGD, exclure PD |
| 19 | `MAINTENANCE_AND_PRESERVATION_COSTS` | Numérique | Frais de maintenance et préservation du bien. | LGD, exclure PD |
| 20 | `TAXES_AND_INSURANCE` | Numérique | Taxes et assurances payées lors du processus de liquidation. | LGD, exclure PD |
| 21 | `MISCELLANEOUS_EXPENSES` | Numérique | Autres frais. | LGD, exclure PD |
| 22 | `ACTUAL_LOSS_CALCULATION` | Numérique | Perte finale calculée. | Cible LGD, exclure PD |
| 23 | `CUMULATIVE_MODIFICATION_COST` | Numérique | Coût cumulé des modifications. | Exclure PD MVP |
| 24 | `INTEREST_RATE_STEP_INDICATOR` | Catégoriel | Indique une modification avec taux par paliers. | Eligible si connu à `t` |
| 25 | `PAYMENT_DEFERRAL_FLAG` | Catégoriel | Indique un report de paiement. | Eligible si connu à `t` |
| 26 | `ESTIMATED_LTV` | Numérique | LTV courant estimé par Freddie Mac. | Eligible à `t` |
| 27 | `ZERO_BALANCE_REMOVAL_UPB` | Numérique | UPB juste avant sortie du portefeuille. | EAD / sortie, exclure PD |
| 28 | `DELINQUENT_ACCRUED_INTEREST` | Numérique | Intérêts de retard accumulés. | Exclure PD MVP |
| 29 | `DELINQUENCY_DUE_TO_DISASTER` | Catégoriel | Retard lié à une catastrophe. | Eligible si connu à `t` |
| 30 | `BORROWER_ASSISTANCE_STATUS_CODE` | Catégoriel | Plan d'assistance emprunteur : forbearance, repayment, trial. | Eligible si connu à `t` |
| 31 | `CURRENT_MONTH_MODIFICATION_COST` | Numérique | Coût de modification du mois courant. | Exclure PD MVP |
| 32 | `INTEREST_BEARING_UPB` | Numérique | Portion du solde courant portant intérêt. | Eligible à `t` |

---

## Variables éligibles sans leakage pour Behavioral PD

Ces variables sont éligibles si elles sont connues à la fin du mois d'observation `t`.
Elles ne sont pas automatiquement candidates finales : elles devront passer les contrôles
de qualité, stabilité et valeur prédictive.

| Groupe | Variables |
|---|---|
| Temps | `MONTHLY_REPORTING_PERIOD`, `LOAN_AGE`, `REMAINING_MONTHS_TO_LEGAL_MATURITY` |
| Encours | `CURRENT_ACTUAL_UPB`, `CURRENT_NON_INTEREST_BEARING_UPB`, `INTEREST_BEARING_UPB` |
| Paiement / retard | `CURRENT_LOAN_DELINQUENCY_STATUS`, `DUE_DATE_OF_LAST_PAID_INSTALLMENT` |
| Conditions courantes | `CURRENT_INTEREST_RATE`, `ESTIMATED_LTV` |
| Restructuration connue à `t` | `MODIFICATION_FLAG`, `INTEREST_RATE_STEP_INDICATOR`, `PAYMENT_DEFERRAL_FLAG` |
| Assistance / contexte | `BORROWER_ASSISTANCE_STATUS_CODE`, `DELINQUENCY_DUE_TO_DISASTER` |

---

## Variables non éligibles comme features PD directes

Ces variables décrivent une sortie finale, une perte, un recouvrement ou une conséquence
du défaut. Elles sont utiles pour les labels, la LGD ou l'EAD, mais ne doivent pas être
utilisées comme variables explicatives directes du modèle PD.

| Groupe | Variables | Raison |
|---|---|---|
| Sortie du portefeuille | `ZERO_BALANCE_CODE`, `ZERO_BALANCE_EFFECTIVE_DATE`, `ZERO_BALANCE_REMOVAL_UPB` | Connaissance de l'événement final |
| Pertes | `ACTUAL_LOSS_CALCULATION`, `DELINQUENT_ACCRUED_INTEREST` | Conséquence du défaut |
| Recouvrements | `MI_RECOVERIES`, `NET_SALE_PROCEEDS`, `NON_MI_RECOVERIES` | Information post-liquidation |
| Frais | `TOTAL_EXPENSES`, `LEGAL_COSTS`, `MAINTENANCE_AND_PRESERVATION_COSTS`, `TAXES_AND_INSURANCE`, `MISCELLANEOUS_EXPENSES` | Frais engagés après défaut / liquidation |
| Coûts de modification | `CUMULATIVE_MODIFICATION_COST`, `CURRENT_MONTH_MODIFICATION_COST` | Peut intégrer une conséquence de restructuration ou de défaut |
| Qualité / règlement | `DEFECT_SETTLEMENT_DATE` | Evénement spécifique non représentatif du comportement emprunteur |

---

## Variable cible PD retenue

Pour prédire le défaut à partir des historiques, la cible doit regarder le futur par
rapport au mois d'observation `t`.

### Définition MVP

```text
BEHAVIORAL_DEFAULT_12M(t) = 1
si le prêt observe un défaut entre t+1 et t+12
```

Un défaut est identifié si au moins une condition future est vraie :

| Condition | Interprétation |
|---|---|
| `CURRENT_LOAN_DELINQUENCY_STATUS >= 3` | 90 jours ou plus de retard |
| `CURRENT_LOAN_DELINQUENCY_STATUS = RA` | REO Acquisition |
| `ZERO_BALANCE_CODE in (02, 03, 09)` | Third Party Sale, Short Sale / Charge Off, REO Disposition |

La variable `CURRENT_LOAN_DELINQUENCY_STATUS` peut être utilisée comme feature au mois
`t`, mais la cible doit être calculée uniquement sur la fenêtre future `t+1` à `t+12`.

---

## Feature Engineering - Behavioral Scorecard

Cette section décrit le plan de construction des features comportementales utilisées
pour le modèle `Behavioral PD`. Toutes les features sont calculées au point
d'observation `t`.

### Prérequis

| Etape | Règle |
|---|---|
| Historique minimum | Travailler uniquement sur les prêts avec un historique supérieur ou égal au seuil défini |
| Jointure | Joindre Performance et Origination via `LOAN_SEQUENCE_NUMBER` |
| Tri | Trier par `LOAN_SEQUENCE_NUMBER`, puis `MONTHLY_REPORTING_PERIOD` croissant |
| Observation | Définir un point d'observation `t` |
| Fenêtre | Définir une fenêtre glissante de `n` mois avant `t` |
| Anti-leakage | Ne jamais utiliser d'information postérieure à `t` |

Le paramètre `n` est à définir pendant le cadrage expérimental. Valeurs suggérées :
`6`, `12` et `24` mois.

---

### Dimension 1 - Sévérité

**Logique métier** : quel est le pire comportement observé ?

Un client qui a touché `DPD3` une seule fois est structurellement différent d'un
client qui n'a jamais dépassé `DPD1`. Le pire incident révèle la capacité maximale
de dégradation.

**Source** : `CURRENT_LOAN_DELINQUENCY_STATUS`

**Feature produite** : `MAX_DPD_n`

```sql
MAX(CURRENT_LOAN_DELINQUENCY_STATUS)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN n PRECEDING AND CURRENT ROW
)
```

**Interprétation** : pire retard atteint sur la fenêtre. Cette feature révèle la
capacité maximale de dégradation du client sur l'historique récent.

---

### Dimension 2 - Fréquence

**Logique métier** : à quelle fréquence le client dérape ?

Un client qui dérape 1 mois sur 24 est différent d'un client qui dérape 10 mois sur
24. La fréquence mesure l'instabilité chronique du comportement. La division par la
longueur de fenêtre garde l'interprétabilité quelle que soit la maturité du prêt.

**Source** : `CURRENT_LOAN_DELINQUENCY_STATUS`

**Features produites** : `RATE_DPD1_n`, `RATE_DPD2_n`

```sql
SUM(CASE WHEN CURRENT_LOAN_DELINQUENCY_STATUS >= 1 THEN 1 ELSE 0 END)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN n PRECEDING AND CURRENT ROW
)
/
COUNT(*)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN n PRECEDING AND CURRENT ROW
)
```

```sql
SUM(CASE WHEN CURRENT_LOAN_DELINQUENCY_STATUS >= 2 THEN 1 ELSE 0 END)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN n PRECEDING AND CURRENT ROW
)
/
COUNT(*)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN n PRECEDING AND CURRENT ROW
)
```

**Interprétation** : proportion de mois en incident sur la fenêtre. La division par
`COUNT(*)` normalise la feature par rapport à la longueur réelle de l'historique
disponible.

---

### Dimension 3 - Récence

**Logique métier** : quand était le dernier incident ?

Un incident vieux de 18 mois avec comportement sain depuis est moins préoccupant
qu'un incident observé il y a 2 mois. La récence capture la dynamique de redressement
ou de dégradation.

**Sources** : `MONTHLY_REPORTING_PERIOD`, `CURRENT_LOAN_DELINQUENCY_STATUS`

**Feature produite** : `MOIS_DEPUIS_DPD1`

```sql
MONTHS_BETWEEN(
    MONTHLY_REPORTING_PERIOD,
    MAX(CASE
        WHEN CURRENT_LOAN_DELINQUENCY_STATUS >= 1
        THEN MONTHLY_REPORTING_PERIOD
    END)
    OVER (
        PARTITION BY LOAN_SEQUENCE_NUMBER
        ORDER BY MONTHLY_REPORTING_PERIOD
        ROWS BETWEEN n PRECEDING AND CURRENT ROW
    )
)
```

Si aucun incident n'est observé dans la fenêtre, la valeur retenue est `n`.

**Interprétation** : plus la valeur est grande, plus l'incident est ancien. Un client
avec une récence élevée et un comportement sain récent est potentiellement en phase
de redressement.

---

### Dimension 4 - Tendance

**Logique métier** : le client s'améliore-t-il ou se dégrade-t-il ?

Un client dont le statut passe de `2` à `0` envoie un signal différent d'un client
qui passe de `0` à `2`. La tendance capture la direction du mouvement, pas seulement
l'état courant.

**Source** : `CURRENT_LOAN_DELINQUENCY_STATUS`

**Features produites** : `TREND_DPD_3M`, `TREND_DPD_6M`

```sql
CURRENT_LOAN_DELINQUENCY_STATUS
- LAG(CURRENT_LOAN_DELINQUENCY_STATUS, 3)
  OVER (
      PARTITION BY LOAN_SEQUENCE_NUMBER
      ORDER BY MONTHLY_REPORTING_PERIOD
  )
```

```sql
CURRENT_LOAN_DELINQUENCY_STATUS
- LAG(CURRENT_LOAN_DELINQUENCY_STATUS, 6)
  OVER (
      PARTITION BY LOAN_SEQUENCE_NUMBER
      ORDER BY MONTHLY_REPORTING_PERIOD
  )
```

**Interprétation** :

| Valeur | Sens |
|---:|---|
| `> 0` | Dégradation |
| `= 0` | Stabilité |
| `< 0` | Redressement |

---

### Dimension 5 - Amortissement

**Logique métier** : le client rembourse-t-il normalement ?

Un client qui rembourse plus vite que prévu montre un signal de solvabilité. Un
client dont l'UPB ne baisse pas, ou augmente via différé, est potentiellement sous
stress. L'écart entre amortissement théorique et réel mesure la santé financière
concrète.

**Sources** : `CURRENT_ACTUAL_UPB`, `UPB`, `LOAN_AGE`, `ORIGINAL_INTEREST_RATE`,
`ORIGINAL_LOAN_TERM`

**Features produites** : `RATIO_UPB`, `ECART_AMORTISSEMENT`

```sql
CURRENT_ACTUAL_UPB / UPB
```

```text
UPB_THEORIQUE(t) = UPB * facteur_amortissement_restant(
    ORIGINAL_INTEREST_RATE,
    ORIGINAL_LOAN_TERM,
    LOAN_AGE
)

ECART_AMORTISSEMENT = UPB_THEORIQUE(t) - CURRENT_ACTUAL_UPB
```

L'UPB théorique se calcule avec le taux et la durée d'origination, selon la formule
d'un prêt amortissable.

**Interprétation** :

| Valeur | Sens |
|---:|---|
| `ECART_AMORTISSEMENT > 0` | Le client rembourse plus vite que prévu |
| `ECART_AMORTISSEMENT = 0` | Le client suit l'amortissement théorique |
| `ECART_AMORTISSEMENT < 0` | Le client est en retard sur son amortissement |

Cette dimension mesure la santé financière concrète au-delà du simple statut de
paiement.

---

### Dimension 6 - Restructuration

**Logique métier** : le client a-t-il eu besoin d'aide ?

Une modification ou un différé signifie que le client n'a pas pu honorer ses
engagements contractuels sans aide externe. C'est un signal de fragilité structurelle,
même si le client est aujourd'hui à jour.

**Sources** : `MODIFICATION_FLAG`, `PAYMENT_DEFERRAL_FLAG`,
`BORROWER_ASSISTANCE_STATUS_CODE`

**Features produites** : `EVER_MODIFIED`, `EVER_DEFERRED`, `NB_MODIFICATIONS`,
`EVER_ASSISTANCE`

```sql
MAX(CASE WHEN MODIFICATION_FLAG = 'Y' THEN 1 ELSE 0 END)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
```

```sql
MAX(CASE WHEN PAYMENT_DEFERRAL_FLAG = 'Y' THEN 1 ELSE 0 END)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
```

```sql
SUM(CASE WHEN MODIFICATION_FLAG = 'Y' THEN 1 ELSE 0 END)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
```

```sql
MAX(CASE
    WHEN BORROWER_ASSISTANCE_STATUS_CODE IS NOT NULL
         AND BORROWER_ASSISTANCE_STATUS_CODE <> ''
    THEN 1 ELSE 0
END)
OVER (
    PARTITION BY LOAN_SEQUENCE_NUMBER
    ORDER BY MONTHLY_REPORTING_PERIOD
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
```

**Interprétation** : toute restructuration signale que le client n'a pas pu honorer
ses engagements seul. C'est un signal de fragilité structurelle, même si le prêt est
à jour au mois `t`.

---

### Récapitulatif des features produites

| Dimension | Ce qu'elle mesure | Ancrage métier | Features |
|---|---|---|---|
| Sévérité | Pire incident | Character (5C) | `MAX_DPD_n` |
| Fréquence | Instabilité chronique | Character (5C) | `RATE_DPD1_n`, `RATE_DPD2_n` |
| Récence | Dynamique récente | Character (5C) | `MOIS_DEPUIS_DPD1` |
| Tendance | Direction du mouvement | Character (5C) | `TREND_DPD_3M`, `TREND_DPD_6M` |
| Amortissement | Santé financière réelle | Capacity + Capital (5C) | `RATIO_UPB`, `ECART_AMORTISSEMENT` |
| Restructuration | Fragilité structurelle | Capacity (5C) | `EVER_MODIFIED`, `EVER_DEFERRED`, `NB_MODIFICATIONS`, `EVER_ASSISTANCE` |

---

## Features combinées Origination x Performance

Les features d'origination décrivent le profil au moment de la décision. Les features
comportementales décrivent ce qui s'est passé après. Leur combinaison permet de
répondre à une question centrale :

> Le profil initial était-il prédictif du comportement observé ?

Ces features sont des interactions. Elles ne remplacent pas les features individuelles :
elles les complètent.

### Prérequis

| Etape | Règle |
|---|---|
| Features amont | Les features d'origination et les 6 dimensions comportementales sont déjà calculées |
| Jointure | La jointure Origination x Performance est déjà établie via `LOAN_SEQUENCE_NUMBER` |
| Temporalité | Toutes les interactions sont calculées au point d'observation `t` |
| Encodage | Les variables catégorielles doivent être encodées avant interaction |
| Valeurs manquantes | Les codes Freddie Mac doivent être traités avant calcul |

Codes manquants à traiter en amont :

| Variable | Code manquant |
|---|---|
| `CREDIT_SCORE` | `9999` |
| `DTI` | `999` |
| `LTV` | `999` |

---

### Combinaison 1 - Fragilité initiale x Comportement observé

**Source** : `DTI` + `RATE_DPD1_n`

**Feature produite** : `DTI_x_RATE_DPD1`

```sql
DTI * RATE_DPD1_n
```

**Interprétation** : un DTI élevé à l'origination combiné à une fréquence de retard
élevée confirme que la fragilité initiale s'est matérialisée dans le comportement.
Un DTI élevé avec comportement sain nuance au contraire le signal initial.

---

### Combinaison 2 - Exposition initiale x Amortissement réel

**Sources** : `LTV` + `RATIO_UPB`, `ECART_AMORTISSEMENT`

**Features produites** : `LTV_x_RATIO_UPB`, `LTV_x_ECART_AMORT`

```sql
LTV * RATIO_UPB
```

```sql
LTV * ECART_AMORTISSEMENT
```

**Interprétation** : un LTV élevé combiné à un amortissement lent signifie que le
client reste fortement exposé. C'est un signal important sur la récupérabilité en
cas de défaut.

---

### Combinaison 3 - Qualité de crédit initiale x Sévérité observée

**Source** : `CREDIT_SCORE` + `MAX_DPD_n`

**Feature produite** : `CREDIT_SCORE_x_MAX_DPD`

```sql
CREDIT_SCORE * MAX_DPD_n
```

**Attention** : `CREDIT_SCORE = 9999` indique une valeur manquante. Ces observations
doivent être exclues du calcul ou imputées avant interaction.

**Interprétation** : un bon score FICO à l'origination avec une sévérité élevée
observée signale une dégradation inattendue. Un mauvais score avec comportement sain
peut indiquer une sous-estimation initiale du client.

---

### Combinaison 4 - Type de produit x Tendance

**Source** : `PRODUCT_TYPE` + `TREND_DPD_3M`

**Feature produite** : `IS_ARM_x_TREND_DPD`

```sql
IS_ARM = CASE
    WHEN PRODUCT_TYPE = 'ARM' THEN 1
    ELSE 0
END

IS_ARM * TREND_DPD_3M
```

**Attention** : `PRODUCT_TYPE` est catégoriel. Il faut l'encoder avant toute
multiplication. Ne pas interagir directement une variable catégorielle brute avec
une variable numérique.

**Interprétation** : un prêt `ARM` en tendance de dégradation est un signal
structurel fort. Le produit peut amplifier la vulnérabilité comportementale.

---

### Combinaison 5 - Statut d'occupation x Restructuration

**Source** : `OCCUPANCY_STATUS` + `EVER_MODIFIED`

**Feature produite** : `IS_INVESTOR_x_EVER_MODIFIED`

```sql
IS_INVESTOR = CASE
    WHEN OCCUPANCY_STATUS = 'I' THEN 1
    ELSE 0
END

IS_INVESTOR * EVER_MODIFIED
```

**Interprétation** : un investisseur ayant déjà été restructuré est un profil de
risque nettement différent d'un propriétaire occupant dans la même situation.

---

### Récapitulatif des interactions produites

| Feature | Type | Valeurs attendues | Ce qu'elle capture |
|---|---|---|---|
| `DTI_x_RATE_DPD1` | Numérique | `[0, ~65]` | Fragilité initiale matérialisée |
| `LTV_x_RATIO_UPB` | Numérique | `[0, ~200]` | Exposition résiduelle réelle |
| `LTV_x_ECART_AMORT` | Numérique signé | Positif = rembourse vite, négatif = retard | Récupérabilité et amortissement réel |
| `CREDIT_SCORE_x_MAX_DPD` | Numérique | `[0, ~8500]` hors manquants | Dégradation inattendue |
| `IS_ARM_x_TREND_DPD` | Numérique signé | `0` si FRM | Vulnérabilité produit |
| `IS_INVESTOR_x_EVER_MODIFIED` | Binaire | `0` ou `1` | Profil de risque composite |

### Note de sélection

Privilégier les interactions avec un sens métier clair pour garder l'interprétabilité
du modèle. Leur pertinence finale sera validée pendant l'étape de sélection des
features.

---

## Note équipe dev

Toutes ces features sont calculées au point d'observation `t`. Aucune information
postérieure à `t` ne doit entrer dans leur construction.

Le calcul de `CURRENT_LOAN_DELINQUENCY_STATUS` doit traiter les valeurs non numériques
comme `RA` avant les agrégations. Pour les features comportementales, `RA` doit être
considéré comme un état de défaut sévère et non comme une valeur numérique ordinaire.

Encoder toutes les variables catégorielles avant de construire les interactions.
Vérifier les valeurs manquantes sur les variables d'origination avant tout calcul :
les codes `9999`, `999` et valeurs équivalentes doivent être traités dans le pipeline
de preprocessing.

---


*Dictionnaire établi sur la base du Freddie Mac Single Family Loan-Level Dataset User Guide.*
