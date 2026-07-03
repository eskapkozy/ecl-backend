CREATE TABLE IF NOT EXISTS loans_performance (
    LOAN_SEQUENCE_NUMBER                        VARCHAR(20) NOT NULL,
    MONTHLY_REPORTING_PERIOD                    DATE NOT NULL,
    CURRENT_ACTUAL_UPB                          NUMERIC(15,2),
    CURRENT_LOAN_DELINQUENCY_STATUS             VARCHAR(5),
    LOAN_AGE                                    SMALLINT,
    REMAINING_MONTHS_TO_LEGAL_MATURITY          SMALLINT,
    DEFECT_SETTLEMENT_DATE                      DATE,
    MODIFICATION_FLAG                           CHAR(1),
    ZERO_BALANCE_CODE                           SMALLINT,
    ZERO_BALANCE_EFFECTIVE_DATE                 DATE,
    CURRENT_INTEREST_RATE                       NUMERIC(6,3),
    CURRENT_NON_INTEREST_BEARING_UPB            NUMERIC(15,2),
    DUE_DATE_OF_LAST_PAID_INSTALLMENT           DATE,
    MI_RECOVERIES                               NUMERIC(15,2),
    NET_SALE_PROCEEDS                           NUMERIC(15,2),
    NON_MI_RECOVERIES                           NUMERIC(15,2),
    TOTAL_EXPENSES                              NUMERIC(15,2),
    LEGAL_COSTS                                 NUMERIC(15,2),
    MAINTENANCE_AND_PRESERVATION_COSTS          NUMERIC(15,2),
    TAXES_AND_INSURANCE                         NUMERIC(15,2),
    MISCELLANEOUS_EXPENSES                      NUMERIC(15,2),
    ACTUAL_LOSS_CALCULATION                     NUMERIC(15,2),
    CUMULATIVE_MODIFICATION_COST                NUMERIC(15,2),
    INTEREST_RATE_STEP_INDICATOR                CHAR(1),
    PAYMENT_DEFERRAL_FLAG                       CHAR(1),
    ESTIMATED_LTV                               NUMERIC(6,2),
    ZERO_BALANCE_REMOVAL_UPB                    NUMERIC(15,2),
    DELINQUENT_ACCRUED_INTEREST                 NUMERIC(15,2),
    DELINQUENCY_DUE_TO_DISASTER                 CHAR(1),
    BORROWER_ASSISTANCE_STATUS_CODE             CHAR(1),
    CURRENT_MONTH_MODIFICATION_COST             NUMERIC(15,2),
    INTEREST_BEARING_UPB                        NUMERIC(15,2),

    PRIMARY KEY (LOAN_SEQUENCE_NUMBER, MONTHLY_REPORTING_PERIOD)
);



CREATE TABLE IF NOT EXISTS loans_origination (
    LOAN_SEQUENCE_NUMBER            VARCHAR(20) PRIMARY KEY,
    CREDIT_SCORE                    NUMERIC(6,1),
    FIRST_TIME_HOMEBUYER_FLAG       CHAR(1),
    MSA                             VARCHAR(10),
    MI_PERCENTAGE                   NUMERIC(5,2),
    NUMBER_OF_UNITS                 SMALLINT,
    OCCUPANCY_STATUS                CHAR(1),
    OCLTV                           NUMERIC(6,2),
    DTI                             NUMERIC(5,2),
    ORIGINAL_UPB                    NUMERIC(15,2),
    LTV                             NUMERIC(6,2),
    ORIGINAL_INTEREST_RATE          NUMERIC(6,3),
    CHANNEL                         CHAR(1),
    PPM_FLAG                        CHAR(1),
    PRODUCT_TYPE                    VARCHAR(5),
    STATE                  CHAR(2),
    PROPERTY_TYPE                   VARCHAR(5),
    POSTAL_CODE                     VARCHAR(10),
    LOAN_PURPOSE                    CHAR(1),
    ORIGINAL_LOAN_TERM               SMALLINT,
    NUMBER_OF_BORROWERS             NUMERIC(5,1),
    SELLER_NAME                     VARCHAR(100),
    SERVICER_NAME                   VARCHAR(100),
    SUPER_CONFORMING_FLAG           CHAR(1),
    PRE_RELIEF_REFI_LOAN_SEQ        VARCHAR(20),
    PROGRAM_INDICATOR               SMALLINT,
    RELIEF_REFINANCE_INDICATOR      CHAR(1),
    PROPERTY_VALUATION_METHOD       SMALLINT,
    IO_FLAG                         CHAR(1),
    MORTGAGE_INSURANCE_CANCELLATION SMALLINT,
    IS_MISSING_CREDIT_SCORE         SMALLINT,
    IS_MISSING_DTI                  SMALLINT
);

CREATE TABLE IF NOT EXISTS loans_default (
    LOAN_SEQUENCE_NUMBER   VARCHAR(20) PRIMARY KEY,
    DEFAULT_FLAG           SMALLINT NOT NULL,
    DEFAULT_TYPE           VARCHAR(20),
    COMPUTED_AT            TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (LOAN_SEQUENCE_NUMBER)
        REFERENCES loans_origination(LOAN_SEQUENCE_NUMBER)
);

CREATE INDEX IF NOT EXISTS idx_perf_loan ON loans_performance (LOAN_SEQUENCE_NUMBER);
CREATE INDEX IF NOT EXISTS idx_default_flag ON loans_default(DEFAULT_FLAG);