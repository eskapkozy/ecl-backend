"""
Feature Engineering — Origination
====================================
Approche : Question-driven feature engineering — cadre des 5C du crédit
Responsabilité : construction des features statiques depuis la table origination

Les données d'origination sont supposées déjà nettoyées.
Aucune gestion de fenêtre — une ligne par prêt en entrée et en sortie.

Colonnes requises :
    - LOAN_SEQUENCE_NUMBER
    - CREDIT_SCORE
    - FIRST_TIME_HOMEBUYER_FLAG
    - NUMBER_OF_BORROWERS
    - DTI
    - LTV
    - OCLTV
    - MI_PERCENTAGE
    - UPB
    - ORIGINAL_LOAN_TERM
    - ORIGINAL_INTEREST_RATE
    - OCCUPANCY_STATUS
    - PROPERTY_TYPE
    - NUMBER_OF_UNITS
    - PRODUCT_TYPE
    - IO_FLAG
    - LOAN_PURPOSE
    - PPM_FLAG
"""

import pandas as pd
import numpy as np


class OriginationFeatures:
    """
    Construit les features statiques d'origination selon les 5C du crédit.

    Usage
    -----
    from origination_features import OriginationFeatures

    features = OriginationFeatures(df_orig).build()
    """

    LOAN_COL = "LOAN_SEQUENCE_NUMBER"

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._cast_numerics()

    def _cast_numerics(self):
        for col in ["CREDIT_SCORE", "DTI", "LTV", "OCLTV",
                    "MI_PERCENTAGE", "UPB", "ORIGINAL_LOAN_TERM",
                    "ORIGINAL_INTEREST_RATE", "NUMBER_OF_BORROWERS",
                    "NUMBER_OF_UNITS"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    # ------------------------------------------------------------------
    # C1 — Capacity
    # Question : le client a-t-il la capacité de rembourser ?
    # ------------------------------------------------------------------

    def _capacity(self) -> pd.DataFrame:
        df = self.df[[self.LOAN_COL]].copy()

        # Mensualité implicite à l'origination
        # Signal : charge absolue du remboursement
        df["mensualite_implicite"] = (
            self.df["UPB"] / self.df["ORIGINAL_LOAN_TERM"]
        )

        # Pression taux sur la durée
        # Signal : coût total du crédit rapporté à la durée
        df["charge_taux_duree"] = (
            self.df["ORIGINAL_INTEREST_RATE"] * self.df["ORIGINAL_LOAN_TERM"]
        )

        return df

    # ------------------------------------------------------------------
    # C2 — Capital
    # Question : quelle est l'exposition et le levier du client ?
    # ------------------------------------------------------------------

    def _capital(self) -> pd.DataFrame:
        df = self.df[[self.LOAN_COL]].copy()

        # Pression levier combinée
        # Signal : surexposé ET surendetté simultanément
        df["pression_levier"] = self.df["LTV"] * self.df["DTI"]

        # Écart LTV / OCLTV
        # Signal : dette senior cachée derrière le prêt principal
        df["ecart_ltv_ocltv"] = self.df["OCLTV"] - self.df["LTV"]

        # Couverture assurance hypothécaire
        # Signal : présence de MI = profil jugé risqué à l'origination
        df["couverture_mi"] = self.df["MI_PERCENTAGE"].fillna(0)

        return df

    # ------------------------------------------------------------------
    # C3 — Collateral
    # Question : quelle est la qualité de la garantie ?
    # ------------------------------------------------------------------

    def _collateral(self) -> pd.DataFrame:
        df = self.df[[self.LOAN_COL]].copy()

        # Bien multi-unité
        # Signal : investissement locatif = profil de risque différent
        df["multi_unite"] = (self.df["NUMBER_OF_UNITS"] > 1).astype(int)

        # Encodage ordinal du statut d'occupation
        # Signal : résidence principale < résidence secondaire < investissement
        occupancy_map = {"P": 0, "S": 1, "I": 2}
        df["occupancy_risk"] = (
            self.df["OCCUPANCY_STATUS"]
            .map(occupancy_map)
            .fillna(-1)
            .astype(int)
        )

        return df

    # ------------------------------------------------------------------
    # C4 — Conditions
    # Question : le produit est-il structurellement risqué ?
    # ------------------------------------------------------------------

    def _conditions(self) -> pd.DataFrame:
        df = self.df[[self.LOAN_COL]].copy()

        # Produit à risque structurel
        # Signal : IO ou ARM = exposition plus forte à la variation de taux
        io_flag      = self.df.get("IO_FLAG", pd.Series("N", index=self.df.index))
        product_type = self.df.get("PRODUCT_TYPE", pd.Series("FRM", index=self.df.index))

        df["produit_risque"] = (
            (io_flag == "Y") | (product_type == "ARM")
        ).astype(int)

        # Flag refinancement
        # Signal : comportement de refinancement = instabilité potentielle
        df["refi_flag"] = (
            self.df["LOAN_PURPOSE"]
            .isin(["R", "C"])  # Refinance / Cash-out Refinance
            .astype(int)
        )

        return df

    # ------------------------------------------------------------------
    # C5 — Character
    # Question : quel est le profil comportemental de l'emprunteur ?
    # ------------------------------------------------------------------

    def _character(self) -> pd.DataFrame:
        df = self.df[[self.LOAN_COL]].copy()

        # Segmentation du score de crédit
        # Signal : subprime / near-prime / prime
        df["credit_segment"] = pd.cut(
            self.df["CREDIT_SCORE"],
            bins=[0, 619, 699, 850],
            labels=[0, 1, 2],   # 0=subprime / 1=near-prime / 2=prime
            right=True
        ).astype(float)

        # Primo-accédant
        # Signal : moins d'expérience de gestion du crédit immobilier
        df["primo_accedant"] = (
            self.df["FIRST_TIME_HOMEBUYER_FLAG"]
            .map({"Y": 1, "N": 0})
            .fillna(-1)
            .astype(int)
        )

        # Co-emprunteur
        # Signal : revenu partagé = moindre risque de défaut
        df["co_emprunteur"] = (
            (self.df["NUMBER_OF_BORROWERS"] > 1).astype(int)
        )

        return df

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Retourne une table — une ligne par prêt —
        avec toutes les features d'origination organisées par dimension 5C.
        """
        frames = [
            self._capacity(),
            self._capital().drop(columns=self.LOAN_COL),
            self._collateral().drop(columns=self.LOAN_COL),
            self._conditions().drop(columns=self.LOAN_COL),
            self._character().drop(columns=self.LOAN_COL),
        ]

        return pd.concat(frames, axis=1)


# ----------------------------------------------------------------------
# Exemple d'usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # df_orig est votre table origination déjà nettoyée
    # features = OriginationFeatures(df_orig).build()
    # print(features.head())
    pass