# warehouse/warehouse.py
from io import StringIO

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text

from warehouse.models import Base


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
        print(hist.columns.tolist())
        hist["REMAINING_MONTHS_TO_LEGAL_MATURITY"] = hist["REMAINING_MONTHS_TO_LEGAL_MATURITY"].fillna(-1).astype(int)
        hist["ZERO_BALANCE_EFFECTIVE_DATE"] = pd.to_datetime(hist["ZERO_BALANCE_EFFECTIVE_DATE"], errors="coerce")
        hist["DUE_DATE_OF_LAST_PAID_INSTALLMENT"] = pd.to_datetime(hist["DUE_DATE_OF_LAST_PAID_INSTALLMENT"],
                                                                   errors="coerce")
        return hist

    def _prepare_orig(self, orig: pd.DataFrame) -> pd.DataFrame:
        orig = orig.copy()
        print(orig.columns.tolist())
        orig["CREDIT_SCORE"] = pd.to_numeric(orig["CREDIT_SCORE"], errors="coerce").fillna(-1).astype(int)
        orig["NUMBER_OF_BORROWERS"] = orig["NUMBER_OF_BORROWERS"].fillna(-1).astype(int)
        orig["CREDIT_SCORE"] = orig["CREDIT_SCORE"].fillna(-1).astype(int)
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
        query_hist = text("SELECT * FROM loans_performance WHERE LOAN_SEQUENCE_NUMBER = :loan_id")
        query_orig = text("SELECT * FROM loans_origination WHERE LOAN_SEQUENCE_NUMBER = :loan_id")
        with self.engine.connect() as conn:
            hist = pd.read_sql(query_hist, conn, params={"loan_id": loan_id})
            orig = pd.read_sql(query_orig, conn, params={"loan_id": loan_id})
        return hist, orig

    def fetch_many(self, loan_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        query_hist = text("SELECT * FROM loans_performance WHERE LOAN_SEQUENCE_NUMBER = ANY(:ids)")
        query_orig = text("SELECT * FROM loans_origination WHERE LOAN_SEQUENCE_NUMBER = ANY(:ids)")
        with self.engine.connect() as conn:
            hist = pd.read_sql(query_hist, conn, params={"ids": loan_ids})
            orig = pd.read_sql(query_orig, conn, params={"ids": loan_ids})
        return hist, orig