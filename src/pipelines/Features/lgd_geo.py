"""
Feature Engineering — Groupe Géographie & Type de bien (GeoFeatures)
========================================================================
Spécifique LGD — seules features non couvertes par OriginationFeatures.
Géographie et type structurel du bien, non redondants avec 5C existants.
"""

import pandas as pd


class GeoFeatures:

    LOAN_COL = "LOAN_SEQUENCE_NUMBER"
    PROPERTY_TYPE_COL = "PROPERTY_TYPE"
    STATE_COL = "STATE"

    def __init__(self, orig_df: pd.DataFrame):
        self.df = orig_df.copy()

    def build(self) -> pd.DataFrame:
        return self.df[[
            self.LOAN_COL, self.PROPERTY_TYPE_COL, self.STATE_COL
        ]].rename(columns={
            self.PROPERTY_TYPE_COL: "property_type",
            self.STATE_COL: "property_state",
        })