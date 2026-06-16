import pandas as pd
import numpy as np


class OriginationFeatures:
    """
    Construit les features statiques d'origination selon les 5C du crédit.
    """

    LOAN_COL = "LOAN_SEQUENCE_NUMBER"

    NUMERIC_COLS = [
        "CREDIT_SCORE",
        "DTI",
        "LTV",
        "OCLTV",
        "MI_PERCENTAGE",
        "ORIGINAL_UPB",
        "ORIGINAL_LOAN_TERM",
        "ORIGINAL_INTEREST_RATE",
        "NUMBER_OF_BORROWERS",
        "NUMBER_OF_UNITS",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._cast_numerics()

    def _cast_numerics(self):
        # VECTORISÉ
        cols = [c for c in self.NUMERIC_COLS if c in self.df.columns]

        if cols:
            self.df[cols] = self.df[cols].apply(
                pd.to_numeric,
                errors="coerce"
            )

    # ------------------------------------------------------------------
    # C1 — Capacity
    # ------------------------------------------------------------------

    def _capacity(self) -> pd.DataFrame:
        # VECTORISÉ
        return pd.DataFrame({
            self.LOAN_COL:
                self.df[self.LOAN_COL],

            "mensualite_implicite":
                self.df["ORIGINAL_UPB"]
                .div(self.df["ORIGINAL_LOAN_TERM"]),

            "charge_taux_duree":
                self.df["ORIGINAL_INTEREST_RATE"]
                .mul(self.df["ORIGINAL_LOAN_TERM"])
        })

    # ------------------------------------------------------------------
    # C2 — Capital
    # ------------------------------------------------------------------

    def _capital(self) -> pd.DataFrame:
        # VECTORISÉ
        return pd.DataFrame({
            self.LOAN_COL:
                self.df[self.LOAN_COL],

            "pression_levier":
                self.df["LTV"]
                .mul(self.df["DTI"]),

            "ecart_ltv_ocltv":
                self.df["OCLTV"]
                .sub(self.df["LTV"]),

            "couverture_mi":
                self.df["MI_PERCENTAGE"]
                .fillna(0)
        })

    # ------------------------------------------------------------------
    # C3 — Collateral
    # ------------------------------------------------------------------

    def _collateral(self) -> pd.DataFrame:
        occupancy_map = {
            "P": 0,
            "S": 1,
            "I": 2
        }

        # VECTORISÉ
        return pd.DataFrame({
            self.LOAN_COL:
                self.df[self.LOAN_COL],

            "multi_unite":
                self.df["NUMBER_OF_UNITS"]
                .gt(1)
                .astype(np.int8),

            "occupancy_risk":
                (
                    self.df["OCCUPANCY_STATUS"]
                    .map(occupancy_map)
                    .fillna(-1)
                    .astype(np.int8)
                )
        })

    # ------------------------------------------------------------------
    # C4 — Conditions
    # ------------------------------------------------------------------

    def _conditions(self) -> pd.DataFrame:

        io_flag = self.df.get(
            "IO_FLAG",
            "N"
        )

        product_type = self.df.get(
            "PRODUCT_TYPE",
            "FRM"
        )

        # VECTORISÉ
        return pd.DataFrame({
            self.LOAN_COL:
                self.df[self.LOAN_COL],

            "produit_risque":
                (
                    (io_flag == "Y")
                    |
                    (product_type == "ARM")
                ).astype(np.int8),

            "refi_flag":
                self.df["LOAN_PURPOSE"]
                .isin(["R", "C"])
                .astype(np.int8)
        })

    # ------------------------------------------------------------------
    # C5 — Character
    # ------------------------------------------------------------------

    def _character(self) -> pd.DataFrame:

        score = self.df["CREDIT_SCORE"]

        # VECTORISÉ
        credit_segment = np.select(
            [
                score.between(1, 619),
                score.between(620, 699),
                score.between(700, 850)
            ],
            [
                0.0,
                1.0,
                2.0
            ],
            default=np.nan
        )

        return pd.DataFrame({
            self.LOAN_COL:
                self.df[self.LOAN_COL],

            "credit_segment":
                credit_segment,

            "primo_accedant":
                (
                    self.df[
                        "FIRST_TIME_HOMEBUYER_FLAG"
                    ]
                    .map({
                        "Y": 1,
                        "N": 0
                    })
                    .fillna(-1)
                    .astype(np.int8)
                ),

            "co_emprunteur":
                (
                    self.df[
                        "NUMBER_OF_BORROWERS"
                    ]
                    .gt(1)
                    .astype(np.int8)
                )
        })

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Retourne une table — une ligne par prêt —
        avec toutes les features d'origination organisées par dimension 5C.
        """

        # VECTORISÉ
        capacity = self._capacity()

        return capacity.join([
            self._capital().drop(columns=self.LOAN_COL),
            self._collateral().drop(columns=self.LOAN_COL),
            self._conditions().drop(columns=self.LOAN_COL),
            self._character().drop(columns=self.LOAN_COL),
        ])


if __name__ == "__main__":
    pass