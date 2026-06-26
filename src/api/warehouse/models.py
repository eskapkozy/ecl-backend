# warehouse/models.py
from sqlalchemy import (
    Column, String, Numeric, SmallInteger, Date, CHAR, Index
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class LoanPerformance(Base):
    __tablename__ = "loans_performance"

    LOAN_SEQUENCE_NUMBER                = Column(String(20),  primary_key=True)
    MONTHLY_REPORTING_PERIOD            = Column(Date,         primary_key=True)
    CURRENT_ACTUAL_UPB                  = Column(Numeric(15, 2))
    CURRENT_LOAN_DELINQUENCY_STATUS     = Column(String(5))
    LOAN_AGE                            = Column(SmallInteger)
    REMAINING_MONTHS_TO_LEGAL_MATURITY  = Column(SmallInteger)
    MODIFICATION_FLAG                   = Column(CHAR(1))
    ZERO_BALANCE_CODE                   = Column(SmallInteger)
    ZERO_BALANCE_EFFECTIVE_DATE         = Column(Date)
    CURRENT_INTEREST_RATE               = Column(Numeric(6, 3))
    CURRENT_NON_INTEREST_BEARING_UPB    = Column(Numeric(15, 2))
    DUE_DATE_OF_LAST_PAID_INSTALLMENT   = Column(Date)
    INTEREST_RATE_STEP_INDICATOR        = Column(CHAR(1))
    ESTIMATED_LTV                       = Column(Numeric(6, 2))
    DELINQUENCY_DUE_TO_DISASTER         = Column(CHAR(1))
    BORROWER_ASSISTANCE_STATUS_CODE     = Column(CHAR(1))
    INTEREST_BEARING_UPB                = Column(Numeric(15, 2))


class LoanOrigination(Base):
    __tablename__ = "loans_origination"

    LOAN_SEQUENCE_NUMBER            = Column(String(20),  primary_key=True)
    CREDIT_SCORE                    = Column(Numeric(6, 1))
    FIRST_TIME_HOMEBUYER_FLAG       = Column(CHAR(1))
    MSA                             = Column(String(10))
    MI_PERCENTAGE                   = Column(Numeric(5, 2))
    NUMBER_OF_UNITS                 = Column(SmallInteger)
    OCCUPANCY_STATUS                = Column(CHAR(1))
    OCLTV                           = Column(Numeric(6, 2))
    DTI                             = Column(Numeric(5, 2))
    ORIGINAL_UPB                    = Column(Numeric(15, 2))
    LTV                             = Column(Numeric(6, 2))
    ORIGINAL_INTEREST_RATE          = Column(Numeric(6, 3))
    CHANNEL                         = Column(CHAR(1))
    PPM_FLAG                        = Column(CHAR(1))
    PRODUCT_TYPE                    = Column(String(5))
    STATE                           = Column(CHAR(2))
    PROPERTY_TYPE                   = Column(String(5))
    POSTAL_CODE                     = Column(String(10))
    LOAN_PURPOSE                    = Column(CHAR(1))
    ORIGINAL_LOAN_TERM              = Column(SmallInteger)
    NUMBER_OF_BORROWERS             = Column(SmallInteger)
    SELLER_NAME                     = Column(String(100))
    SERVICER_NAME                   = Column(String(100))
    SUPER_CONFORMING_FLAG           = Column(CHAR(1))
    PRE_RELIEF_REFI_LOAN_SEQ        = Column(String(20))
    PROGRAM_INDICATOR               = Column(SmallInteger)
    RELIEF_REFINANCE_INDICATOR      = Column(CHAR(1))
    PROPERTY_VALUATION_METHOD       = Column(SmallInteger)
    IO_FLAG                         = Column(CHAR(1))
    MORTGAGE_INSURANCE_CANCELLATION = Column(SmallInteger)
    IS_MISSING_CREDIT_SCORE         = Column(SmallInteger)
    IS_MISSING_DTI                  = Column(SmallInteger)