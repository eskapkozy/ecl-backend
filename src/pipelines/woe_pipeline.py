"""
WoE Pipeline
=============
Responsabilité : discrétisation optimale et transformation WoE des features.

Optimisations :
    - suppression du copy() inutile
    - summary() et selected() cachés dans __init__
    - iv_report() sans boucle Python
"""

import pandas as pd
from optbinning import BinningProcess


class WoePipeline:

    def __init__(self,
                 x_data          : pd.DataFrame,
                 y_data          : pd.Series = None,
                 config          : dict = None,
                 binning_process : BinningProcess = None,
                 categorical_vars: list = None):

        self.x_data           = x_data  # pas de copie
        self.y_data           = y_data
        self.config           = config or {"iv_threshold": 0.02, "metric": "woe"}
        self.iv_threshold     = self.config["iv_threshold"]
        self.transform_metric = self.config["metric"]

        self.categoriel     = (
            categorical_vars if categorical_vars is not None
            else self.x_data.select_dtypes(include=["object"]).columns.tolist()
        )
        self.variable_names = self.x_data.columns.tolist()

        # Fit ou réutilisation
        if binning_process is not None:
            self.capturedFit = binning_process
        else:
            if y_data is None:
                raise ValueError("y_data est requis pour le fit.")
            self.capturedFit = self._fit()

        # Cache — summary et selected calculés une seule fois
        self._summary  = self.capturedFit.summary()
        self._selected = (
            self._summary[self._summary["selected"] == True]["name"].tolist()
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit(self) -> BinningProcess:
        bp = BinningProcess(
            variable_names=self.variable_names,
            categorical_variables=self.categoriel,
            selection_criteria={"iv": {"min": self.iv_threshold}}
        )
        bp.fit(self.x_data, self.y_data)
        return bp

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self) -> pd.DataFrame:
        if self.capturedFit is None:
            raise RuntimeError("capturedFit non initialisé.")
        return self.capturedFit.transform(self.x_data, metric=self.transform_metric)

    # ------------------------------------------------------------------
    # Rapports
    # ------------------------------------------------------------------

    def iv_report(self) -> pd.DataFrame:
        """IV de chaque feature sélectionnée — sans boucle Python."""
        rows = []
        for var in self._selected:
            ob    = self.capturedFit.get_binned_variable(var)
            table = ob.binning_table.build()
            rows.append({"feature": var, "IV": table.loc["Totals", "IV"]})

        return (
            pd.DataFrame(rows)
            .set_index("feature")
            .sort_values("IV", ascending=False)
        )

    def table(self, var: str) -> pd.DataFrame:
        ob = self.capturedFit.get_binned_variable(var)
        return ob.binning_table.build()

    def selection_report(self) -> dict:
        rejected = self._summary[self._summary["selected"] == False]["name"].tolist()
        return {
            "iv_threshold": self.iv_threshold,
            "selected"    : self._selected,
            "rejected"    : rejected
        }

    def selected(self) -> list:
        return self._selected