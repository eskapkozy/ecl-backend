# warehouse/warehouse.py
from io import StringIO

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text

from src.api.warehouse.models import Base


class WarehouseLoader:

    def __init__(self, db_url: str, db_params: dict):
        self.engine    = create_engine(db_url)
        self.db_params = db_params
        Base.metadata.create_all(self.engine)

    def load(self, hist: pd.DataFrame, orig: pd.DataFrame):
        self._copy(self._prepare_hist(hist), "loans_performance")
        self._copy(self._prepare_orig(orig), "loans_origination")

    def _copy(self, df: pd.DataFrame, table: str):
        conn = psycopg2.connect(**self.db_params)
        cur  = conn.cursor()
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cur.copy_expert(f"COPY {table} FROM STDIN WITH CSV", buffer)
        conn.commit()
        cur.close()
        conn.close()
        print(f"{table} : {len(df)} lignes chargées")

    def _prepare_hist(self, hist: pd.DataFrame) -> pd.DataFrame:
        hist = hist.copy()
        hist["REMAINING_MONTHS_TO_LEGAL_MATURITY"] = hist["REMAINING_MONTHS_TO_LEGAL_MATURITY"].fillna(-1).astype(int)
        hist["ZERO_BALANCE_EFFECTIVE_DATE"] = pd.to_datetime(hist["ZERO_BALANCE_EFFECTIVE_DATE"], errors="coerce")
        hist["DUE_DATE_OF_LAST_PAID_INSTALLMENT"] = pd.to_datetime(hist["DUE_DATE_OF_LAST_PAID_INSTALLMENT"],
                                                                   errors="coerce")
        hist["NET_SALE_PROCEEDS"] = pd.to_numeric(hist["NET_SALE_PROCEEDS"], errors="coerce")
        hist["MI_RECOVERIES"] = pd.to_numeric(hist["MI_RECOVERIES"], errors="coerce")
        hist["NON_MI_RECOVERIES"] = pd.to_numeric(hist["NON_MI_RECOVERIES"], errors="coerce")
        hist["DEFECT_SETTLEMENT_DATE"] = pd.to_datetime(
            hist["DEFECT_SETTLEMENT_DATE"], errors="coerce"
        )

        hist["TOTAL_EXPENSES"] = pd.to_numeric( hist["TOTAL_EXPENSES"], errors="coerce")
        hist["LEGAL_COSTS"] = pd.to_numeric(hist["LEGAL_COSTS"], errors="coerce")
        hist["MAINTENANCE_AND_PRESERVATION_COSTS"] = pd.to_numeric(hist["MAINTENANCE_AND_PRESERVATION_COSTS"], errors="coerce")
        hist["TAXES_AND_INSURANCE"] = pd.to_numeric(hist["TAXES_AND_INSURANCE"], errors="coerce")
        hist["MISCELLANEOUS_EXPENSES"] = pd.to_numeric(hist["MISCELLANEOUS_EXPENSES"], errors="coerce")
        hist["ACTUAL_LOSS_CALCULATION"] = pd.to_numeric(hist["ACTUAL_LOSS_CALCULATION"], errors="coerce")
        hist["CUMULATIVE_MODIFICATION_COST"] = pd.to_numeric(hist["CUMULATIVE_MODIFICATION_COST"], errors="coerce")
        hist["ZERO_BALANCE_REMOVAL_UPB"] = pd.to_numeric(hist["ZERO_BALANCE_REMOVAL_UPB"], errors="coerce")
        hist["DELINQUENT_ACCRUED_INTEREST"] = pd.to_numeric(hist["DELINQUENT_ACCRUED_INTEREST"], errors="coerce")
        hist["CURRENT_MONTH_MODIFICATION_COST"] = pd.to_numeric(hist["CURRENT_MONTH_MODIFICATION_COST"], errors="coerce")

        HIST_COL_ORDER = [
            "LOAN_SEQUENCE_NUMBER",
            "MONTHLY_REPORTING_PERIOD",
            "CURRENT_ACTUAL_UPB",
            "CURRENT_LOAN_DELINQUENCY_STATUS",
            "LOAN_AGE",
            "REMAINING_MONTHS_TO_LEGAL_MATURITY",
            "DEFECT_SETTLEMENT_DATE",
            "MODIFICATION_FLAG",
            "ZERO_BALANCE_CODE",
            "ZERO_BALANCE_EFFECTIVE_DATE",
            "CURRENT_INTEREST_RATE",
            "CURRENT_NON_INTEREST_BEARING_UPB",
            "DUE_DATE_OF_LAST_PAID_INSTALLMENT",
            "MI_RECOVERIES",
            "NET_SALE_PROCEEDS",
            "NON_MI_RECOVERIES",
            "TOTAL_EXPENSES",
            "LEGAL_COSTS",
            "MAINTENANCE_AND_PRESERVATION_COSTS",
            "TAXES_AND_INSURANCE",
            "MISCELLANEOUS_EXPENSES",
            "ACTUAL_LOSS_CALCULATION",
            "CUMULATIVE_MODIFICATION_COST",
            "INTEREST_RATE_STEP_INDICATOR",
            "PAYMENT_DEFERRAL_FLAG",
            "ESTIMATED_LTV",
            "ZERO_BALANCE_REMOVAL_UPB",
            "DELINQUENT_ACCRUED_INTEREST",
            "DELINQUENCY_DUE_TO_DISASTER",
            "BORROWER_ASSISTANCE_STATUS_CODE",
            "CURRENT_MONTH_MODIFICATION_COST",
            "INTEREST_BEARING_UPB",
        ]

        return hist[HIST_COL_ORDER]

    def _prepare_orig(self, orig: pd.DataFrame) -> pd.DataFrame:
        orig = orig.copy()
        orig["CREDIT_SCORE"] = pd.to_numeric(orig["CREDIT_SCORE"], errors="coerce").fillna(-1).astype(int)
        orig["NUMBER_OF_BORROWERS"] = orig["NUMBER_OF_BORROWERS"].fillna(-1).astype(int)
        orig["MI_PERCENTAGE"] = orig["MI_PERCENTAGE"].fillna(0)
        orig["POSTAL_CODE"] = orig["POSTAL_CODE"].astype(str)
        orig["PROGRAM_INDICATOR"] = pd.to_numeric(orig["PROGRAM_INDICATOR"], errors="coerce").fillna(-1).astype(int)
        orig["MORTGAGE_INSURANCE_CANCELLATION"] = pd.to_numeric(orig["MORTGAGE_INSURANCE_CANCELLATION"],
                                                                errors="coerce").fillna(-1).astype(int)
        

        ORIG_COL_ORDER = [
            "LOAN_SEQUENCE_NUMBER", "CREDIT_SCORE", "FIRST_TIME_HOMEBUYER_FLAG", "MSA",
            "MI_PERCENTAGE", "NUMBER_OF_UNITS", "OCCUPANCY_STATUS", "OCLTV", "DTI",
            "ORIGINAL_UPB", "LTV", "ORIGINAL_INTEREST_RATE", "CHANNEL", "PPM_FLAG",
            "PRODUCT_TYPE", "STATE", "PROPERTY_TYPE", "POSTAL_CODE", "LOAN_PURPOSE",
            "ORIGINAL_LOAN_TERM", "NUMBER_OF_BORROWERS", "SELLER_NAME", "SERVICER_NAME",
            "SUPER_CONFORMING_FLAG", "PRE_RELIEF_REFI_LOAN_SEQ", "PROGRAM_INDICATOR",
            "RELIEF_REFINANCE_INDICATOR", "PROPERTY_VALUATION_METHOD", "IO_FLAG",
            "MORTGAGE_INSURANCE_CANCELLATION", "IS_MISSING_CREDIT_SCORE", "IS_MISSING_DTI"
        ]

        return orig[ORIG_COL_ORDER]


class WarehouseReader:

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def fetch(self, loan_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        query_hist = text("SELECT * FROM loans_performance WHERE loan_sequence_number = :loan_id")
        query_orig = text("SELECT * FROM loans_origination WHERE loan_sequence_number = :loan_id")
        with self.engine.connect() as conn:
            hist = pd.read_sql(query_hist, conn, params={"loan_id": loan_id})
            orig = pd.read_sql(query_orig, conn, params={"loan_id": loan_id})

        hist.columns = hist.columns.str.upper()
        orig.columns = orig.columns.str.upper()

        return hist, orig

    def fetch_many(self, loan_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        query_hist = text("SELECT * FROM loans_performance WHERE LOAN_SEQUENCE_NUMBER = ANY(:ids)")
        query_orig = text("SELECT * FROM loans_origination WHERE LOAN_SEQUENCE_NUMBER = ANY(:ids)")
        with self.engine.connect() as conn:
            hist = pd.read_sql(query_hist, conn, params={"ids": loan_ids})
            orig = pd.read_sql(query_orig, conn, params={"ids": loan_ids})

        hist.columns = hist.columns.str.upper()
        orig.columns = orig.columns.str.upper()

        return hist, orig