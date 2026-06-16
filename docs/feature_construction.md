# Feature Engineering — Documentation
**Projet** : Outil de Recommandation de Crédit — Gamme ECL (PD / LGD / EAD)
**Étape** : Construction de Features
**Approche** : Question-driven feature engineering
**Date** : Juin 2026

---

## Principes directeurs

Chaque feature répond à une question métier précise. On ne construit pas de variables mécaniquement — on part d'un angle de lecture du comportement, on formule une question, puis on traduit cette question en opération sur les données.

**Règle fondamentale** : toute information postérieure au point d'observation est exclue (data leakage).

---

## Architecture des modules

```
window_builder.py         →  découpe le panel historique en fenêtre de 12 mois
delinquency_features.py   →  features comportementales — Groupe Retard
capital_features.py       →  features comportementales — Groupe Capital
origination_features.py   →  features statiques — Origination (cadre 5C)
```

**Flux d'utilisation**

```
Historique brut
      ↓
window_builder.py         →  df_win (panel 12 mois)
      ↓
delinquency_features.py   →  retard_features (1 ligne / prêt)
capital_features.py       →  capital_features (1 ligne / prêt)
      ↓
origination_features.py   →  orig_features (1 ligne / prêt)
      ↓
Jointure des trois tables sur LOAN_SEQUENCE_NUMBER
      ↓
Table finale agrégée → EDA → Sélection → Modèle
```

---

## Paramètres de modélisation retenus

| Paramètre             | Valeur   | Justification                              |
|-----------------------|----------|--------------------------------------------|
| Fenêtre d'observation | 12 mois  | Standard industrie crédit immobilier       |
| Outcome window        | 12 mois  | Aligné IFRS 9                              |
| Seuil de défaut       | 90 DPD   | Standard réglementaire IFRS 9 / Bâle III   |

---

## Groupe 1 — Retard

**Colonne source** : `CURRENT_LOAN_DELINQUENCY_STATUS` (mois de retard × 30 = DPD jours)

| Feature | Question métier | Opération |
|---|---|---|
| `freq` | Combien de fois le client est-il en retard ? | `count(DPD > 0) / nb_mois_total` |
| `severite` | En moyenne jusqu'où vont les retards ? | `mean(DPD)` |
| `tendance` | Les retards augmentent-ils ou diminuent-ils ? | `slope(DPD)` — régression linéaire |
| `recuperation` | Le client revient-il à 0 DPD après un retard ? | `nb_mois_avant_retour_0` |
| `profondeur_max` | Quel est le pire DPD jamais atteint ? | `max(DPD)` |
| `n_profondeur_max` | Combien de fois ce pire niveau a-t-il été atteint ? | `count(DPD == max)` |

**Features combinées**

| Feature | Composition | Signal |
|---|---|---|
| `freq_x_profondeur_max` | `freq × profondeur_max` | Souvent en retard ET va loin |
| `freq_x_tendance` | `freq × tendance` | Retards fréquents ET croissants |
| `freq_x_recuperation` | `freq × recuperation` | Fréquence vs vitesse de rattrapage |
| `recidivisme_extreme` | `n_profondeur_max × profondeur_max` | Récidivisme au niveau le plus critique |

---

## Groupe 2 — Capital

**Colonnes sources** : `CURRENT_ACTUAL_UPB`, `ORIGINAL_UPB`, `LOAN_AGE`, `REMAINING_MONTHS_TO_LEGAL_MATURITY`, `CURRENT_INTEREST_RATE`

| Feature | Question métier | Opération |
|---|---|---|
| `niveau` | Quelle part du capital reste due ? | `UPB_courant / UPB_origination` |
| `progression` | Le capital baisse-t-il régulièrement ? | `std(delta_UPB)` |
| `ecart_au_plan` | Est-on en retard sur le remboursement théorique ? | `UPB_theorique - UPB_reel` |
| `anticipation` | Rembourse-t-il plus que prévu ? | `nb_mois_sup_plan / nb_mois_total` |

---

## Groupe 3 — Origination (cadre 5C)

### C1 — Capacity
*Question : le client a-t-il la capacité de rembourser ?*

| Feature | Composition | Signal |
|---|---|---|
| `mensualite_implicite` | `UPB / ORIGINAL_LOAN_TERM` | Charge absolue du remboursement |
| `charge_taux_duree` | `ORIGINAL_INTEREST_RATE × ORIGINAL_LOAN_TERM` | Coût total du crédit |

### C2 — Capital
*Question : quelle est l'exposition et le levier du client ?*

| Feature | Composition | Signal |
|---|---|---|
| `pression_levier` | `LTV × DTI` | Surexposé ET surendetté simultanément |
| `ecart_ltv_ocltv` | `OCLTV - LTV` | Dette senior cachée derrière le prêt |
| `couverture_mi` | `MI_PERCENTAGE` | Profil jugé risqué à l'origination |

### C3 — Collateral
*Question : quelle est la qualité de la garantie ?*

| Feature | Composition | Signal |
|---|---|---|
| `multi_unite` | `1 si NUMBER_OF_UNITS > 1` | Investissement locatif |
| `occupancy_risk` | Encodage ordinal `OCCUPANCY_STATUS` | Principal=0 / Second=1 / Invest=2 |

### C4 — Conditions
*Question : le produit est-il structurellement risqué ?*

| Feature | Composition | Signal |
|---|---|---|
| `produit_risque` | `1 si IO_FLAG=Y ou PRODUCT_TYPE=ARM` | Produit à exposition variable |
| `refi_flag` | `1 si LOAN_PURPOSE ∈ {R, C}` | Comportement de refinancement |

### C5 — Character
*Question : quel est le profil comportemental de l'emprunteur ?*

| Feature | Composition | Signal |
|---|---|---|
| `credit_segment` | `CREDIT_SCORE` binné | 0=subprime / 1=near-prime / 2=prime |
| `primo_accedant` | `FIRST_TIME_HOMEBUYER_FLAG` | Moins d'expérience crédit immobilier |
| `co_emprunteur` | `1 si NUMBER_OF_BORROWERS > 1` | Revenu partagé = moindre risque |

---

## Prochaine étape — Sélection de Features

- Ouvrir un notebook EDA dédié
- Analyse univariée → distributions → décision de mise à l'échelle
- Analyse bivariée → pouvoir discriminant de chaque feature vis-à-vis du défaut
- Multicolinéarité → matrice de corrélation + VIF → éviction des features redondantes
- Implémenter les décisions dans `feature_selector.py`

---

*Document versé dans le repo projet — Référentiel feature engineering*
*Prochaine révision : à l'issue de l'EDA*