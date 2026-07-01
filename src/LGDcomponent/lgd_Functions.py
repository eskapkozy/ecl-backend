import pandas as pd
import numpy as np


# =============================================================================
# LGD TARGET COMPUTATION — Freddie Mac Single Family Loan-Level Dataset
# =============================================================================
#
# SOURCE TABLE : loans_performance (hist)
# RELEVANT COLUMNS :
#   - LOAN_SEQUENCE_NUMBER          : identifiant unique du loan
#   - MONTHLY_REPORTING_PERIOD      : date mensuelle de l'observation
#   - CURRENT_ACTUAL_UPB            : solde impayé réel à la date de reporting
#   - CURRENT_LOAN_DELINQUENCY_STATUS : statut de délinquance (en mois de retard)
#   - ZERO_BALANCE_CODE             : code de résolution du loan
#   - NET_SALE_PROCEEDS             : produit net de la vente du collatéral
#   - MI_RECOVERIES                 : recouvrement via assurance hypothécaire (MI)
#   - NON_MI_RECOVERIES             : autres recouvrements (hors MI)
#
# DÉFINITION RÉGLEMENTAIRE (IFRS 9 / Basel III) :
#   LGD = fraction de l'exposition perdue en cas de défaut, nette de tout recouvrement.
#   LGD = 1 - (Recovery / EAD)
#
# COMPOSANTES :
#
#   EAD (Exposure at Default) :
#     Valeur de CURRENT_ACTUAL_UPB au premier mois où le loan atteint
#     le seuil de défaut (CURRENT_LOAN_DELINQUENCY_STATUS >= 3, soit 90 DPD).
#     Représente l'exposition totale au moment de la bascule en défaut.
#
#   Recovery — RÉVISION :
#     ZERO_BALANCE_REMOVAL_UPB a été écarté : cette colonne représente le
#     solde restant dû au moment de la sortie du portefeuille, PAS un
#     montant recouvré. Elle reste proche de l'EAD quel que soit le niveau
#     de recouvrement réel, ce qui produisait un LGD artificiellement
#     proche de 1 pour la quasi-totalité des loans (biais constaté en test).
#
#     Le recouvrement réel est reconstruit à partir des montants effectivement
#     perçus après liquidation du collatéral :
#
#       Recovery = NET_SALE_PROCEEDS + MI_RECOVERIES + NON_MI_RECOVERIES
#
#     NET_SALE_PROCEEDS peut être négatif (frais de vente excédant le produit
#     brut) — conservé tel quel, il contribue logiquement à aggraver la perte.
#
#     Codes ZERO_BALANCE_CODE considérés comme résolution définitive
#     avec liquidation du collatéral :
#       02 = Third Party Sale (foreclosure sale)
#       03 = Short Sale
#       09 = REO Disposition
#       15 = Note Sale
#
#   LGD :
#     LGD = 1 - (Recovery / EAD)
#
#     Interprétation :
#       LGD = 0  → recouvrement total (aucune perte)
#       LGD = 1  → perte totale (aucun recouvrement)
#
#   CLIPPING [0, 1] :
#     Des valeurs hors bornes peuvent survenir (proceeds négatifs extrêmes,
#     recoveries supérieures à l'EAD). On clip strictement dans [0, 1]
#     pour respecter la définition économique.
#
# DATA LEAKAGE :
#   Cette fonction doit être appelée UNIQUEMENT sur les données historiques
#   de la population en défaut DÉJÀ RÉSOLUE (training set), AVANT tout split
#   train/val/test. La discrétisation quantile (K=8 bins) sera fit sur le
#   train uniquement.
#
#   Les loans en défaut NON résolus ne sont pas utilisables pour
#   l'entraînement — ils n'ont pas encore de LGD observé. Ils deviennent
#   uniquement des cibles d'inférence une fois le modèle entraîné.
#
# =============================================================================

# Codes ZERO_BALANCE indiquant une résolution définitive avec liquidation
RESOLUTION_CODES = {2, 3, 9, 15}

# Seuil de défaut : 3 mois de retard = 90 DPD (cohérent avec le modèle PD)
DEFAULT_DPD_THRESHOLD = 3


def compute_lgd_target(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la target LGD pour chaque loan en défaut résolu.

    La fonction opère en trois étapes :
      1. Identifier l'EAD : UPB au premier mois de défaut (DPD >= 90)
      2. Identifier le Recovery réel : NET_SALE_PROCEEDS + MI_RECOVERIES
         + NON_MI_RECOVERIES, au dernier mois avant résolution
      3. Calculer LGD = 1 - (Recovery / EAD), clippé dans [0, 1]

    Args:
        perf_df (pd.DataFrame):
            DataFrame issu de loans_performance (hist complet, non windowé),
            filtré ou non sur les loans en défaut.
            Doit contenir les colonnes :
              - LOAN_SEQUENCE_NUMBER
              - MONTHLY_REPORTING_PERIOD (datetime ou string triable)
              - CURRENT_ACTUAL_UPB (float)
              - CURRENT_LOAN_DELINQUENCY_STATUS (int ou string numérique)
              - ZERO_BALANCE_CODE (float/int, NaN si non résolu)
              - NET_SALE_PROCEEDS (float)
              - MI_RECOVERIES (float)
              - NON_MI_RECOVERIES (float)

    Returns:
        pd.DataFrame avec colonnes :
              - LOAN_SEQUENCE_NUMBER
              - ead                  : exposure at default
              - recovery             : NET_SALE_PROCEEDS + MI_RECOVERIES + NON_MI_RECOVERIES
              - lgd_target           : valeur LGD clippée dans [0, 1]
    """

    df = perf_df.copy()

    # -- Normalisation des types -----------------------------------------------

    df["MONTHLY_REPORTING_PERIOD"] = pd.to_datetime(
        df["MONTHLY_REPORTING_PERIOD"]
    )

    df["CURRENT_LOAN_DELINQUENCY_STATUS"] = pd.to_numeric(
        df["CURRENT_LOAN_DELINQUENCY_STATUS"], errors="coerce"
    )

    df["ZERO_BALANCE_CODE"] = pd.to_numeric(
        df["ZERO_BALANCE_CODE"], errors="coerce"
    )

    for col in ["NET_SALE_PROCEEDS", "MI_RECOVERIES", "NON_MI_RECOVERIES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- Tri chronologique par loan --------------------------------------------

    df = df.sort_values(
        ["LOAN_SEQUENCE_NUMBER", "MONTHLY_REPORTING_PERIOD"]
    )

    # -- ÉTAPE 1 : EAD — UPB au premier mois de défaut (DPD >= 90) -----------

    in_default = df[
        df["CURRENT_LOAN_DELINQUENCY_STATUS"] >= DEFAULT_DPD_THRESHOLD
    ]

    ead_df = (
        in_default
        .groupby("LOAN_SEQUENCE_NUMBER")
        .first()
        .reset_index()
        [["LOAN_SEQUENCE_NUMBER", "CURRENT_ACTUAL_UPB"]]
        .rename(columns={"CURRENT_ACTUAL_UPB": "ead"})
    )

    # -- ÉTAPE 2 : Recovery réel — proceeds + recoveries au mois de résolution -
    #
    # On filtre les observations avec un code de résolution définitive,
    # puis on extrait les montants de recouvrement effectifs.

    resolved = df[
        df["ZERO_BALANCE_CODE"].isin(RESOLUTION_CODES)
    ]

    recovery_df = (
        resolved
        .groupby("LOAN_SEQUENCE_NUMBER")
        .last()
        .reset_index()
    )

    recovery_df["recovery"] = (
        recovery_df["NET_SALE_PROCEEDS"].fillna(0)
        + recovery_df["MI_RECOVERIES"].fillna(0)
        + recovery_df["NON_MI_RECOVERIES"].fillna(0)
    )

    recovery_df = recovery_df[["LOAN_SEQUENCE_NUMBER", "recovery"]]

    # -- ÉTAPE 3 : Calcul LGD -------------------------------------------------
    #
    # Inner join : défaut confirmé ET résolution définitive observée.

    lgd_df = ead_df.merge(recovery_df, on="LOAN_SEQUENCE_NUMBER", how="inner")

    # Garde : EAD = 0 est un artefact de séquence (ex. loan déjà clôturé dont
    # le dernier DPD résiduel affiché atteint encore 3, CURRENT_ACTUAL_UPB
    # retombé à 0). Économiquement non interprétable — exclu avant calcul.
    n_zero_ead = (lgd_df["ead"] == 0).sum()
    if n_zero_ead > 0:
        print(f"[LGD] Loans exclus pour EAD=0 (artefact de séquence) : {n_zero_ead}")
        lgd_df = lgd_df[lgd_df["ead"] > 0].copy()

    lgd_df["lgd_target"] = 1 - (lgd_df["recovery"] / lgd_df["ead"])

    # Clip strict dans [0, 1]
    lgd_df["lgd_target"] = lgd_df["lgd_target"].clip(lower=0.0, upper=1.0)

    # -- Vérification de sanité -----------------------------------------------

    n_total = lgd_df.shape[0]
    n_valid = lgd_df["lgd_target"].notna().sum()
    raw_ratio = 1 - (lgd_df["recovery"] / lgd_df["ead"])
    n_clipped = ((raw_ratio < 0) | (raw_ratio > 1)).sum()

    print(f"[LGD] Loans avec target calculée : {n_valid} / {n_total}")
    print(f"[LGD] Observations clippées hors [0,1] : {n_clipped}")
    print(f"[LGD] Distribution LGD :\n{lgd_df['lgd_target'].describe().round(4)}")
    

    return lgd_df[
        ["LOAN_SEQUENCE_NUMBER", "ead", "recovery", "lgd_target"]
    ]