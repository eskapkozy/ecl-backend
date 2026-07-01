"""
Cleaning Pipeline
==================
Responsabilité : nettoyage et préparation des données brutes
                 avant toute construction de features.

Deux méthodes publiques :
    - orig_impute(orig)  →  nettoyage table origination
    - hist_impute(hist)  →  nettoyage table historique, sélection des colonnes éligibles
"""
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np


class CleaningPipeline:
    logger = logging.getLogger(__name__)



    def __init__(self,hist_path:str= None,orig_path: str = None ,loan_number: str = None, hist : pd.DataFrame = None, orig : pd.DataFrame = None, mode: str = "pd"):



        ORIGINATION_COLUMNS = [
            "CREDIT_SCORE",  # 1
            "FIRST_PAYMENT_DATE",  # 2  — post-décision
            "FIRST_TIME_HOMEBUYER_FLAG",  # 3
            "MATURITY_DATE",  # 4  — post-décision
            "MSA",  # 5
            "MI_PERCENTAGE",  # 6
            "NUMBER_OF_UNITS",  # 7
            "OCCUPANCY_STATUS",  # 8
            "OCLTV",  # 9
            "DTI",  # 10
            "ORIGINAL_UPB",  # 11
            "LTV",  # 12
            "ORIGINAL_INTEREST_RATE",  # 13
            "CHANNEL",  # 14
            "PPM_FLAG",  # 15
            "PRODUCT_TYPE",  # 16
            "STATE",  # 17
            "PROPERTY_TYPE",  # 18
            "POSTAL_CODE",  # 19
            "LOAN_SEQUENCE_NUMBER",  # 20 — post-décision (clé Freddie Mac)
            "LOAN_PURPOSE",  # 21
            "ORIGINAL_LOAN_TERM",  # 22
            "NUMBER_OF_BORROWERS",  # 23
            "SELLER_NAME",  # 24
            "SERVICER_NAME",  # 25
            "SUPER_CONFORMING_FLAG",  # 26
            "PRE_RELIEF_REFI_LOAN_SEQ",  # 27
            "PROGRAM_INDICATOR",  # 28
            "RELIEF_REFINANCE_INDICATOR",  # 29
            "PROPERTY_VALUATION_METHOD",  # 30
            "IO_FLAG",  # 31
            "MORTGAGE_INSURANCE_CANCELLATION",  # 32
        ]



        PERFORMANCE_COLUMNS = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'CURRENT_ACTUAL_UPB',
                               'CURRENT_LOAN_DELINQUENCY_STATUS', 'LOAN_AGE', 'REMAINING_MONTHS_TO_LEGAL_MATURITY',
                               'DEFECT_SETTLEMENT_DATE', 'MODIFICATION_FLAG', 'ZERO_BALANCE_CODE',
                               'ZERO_BALANCE_EFFECTIVE_DATE', 'CURRENT_INTEREST_RATE',
                               'CURRENT_NON_INTEREST_BEARING_UPB', 'DUE_DATE_OF_LAST_PAID_INSTALLMENT', 'MI_RECOVERIES',
                               'NET_SALE_PROCEEDS', 'NON_MI_RECOVERIES', 'TOTAL_EXPENSES', 'LEGAL_COSTS',
                               'MAINTENANCE_AND_PRESERVATION_COSTS', 'TAXES_AND_INSURANCE', 'MISCELLANEOUS_EXPENSES',
                               'ACTUAL_LOSS_CALCULATION', 'CUMULATIVE_MODIFICATION_COST',
                               'INTEREST_RATE_STEP_INDICATOR', 'PAYMENT_DEFERRAL_FLAG', 'ESTIMATED_LTV',
                               'ZERO_BALANCE_REMOVAL_UPB', 'DELINQUENT_ACCRUED_INTEREST', 'DELINQUENCY_DUE_TO_DISASTER',
                               'BORROWER_ASSISTANCE_STATUS_CODE', 'CURRENT_MONTH_MODIFICATION_COST',
                               'INTEREST_BEARING_UPB']


        self.ELIGIBLE_COLUMNS = [
            "LOAN_SEQUENCE_NUMBER",
            "MONTHLY_REPORTING_PERIOD",
            "CURRENT_ACTUAL_UPB",
            "CURRENT_LOAN_DELINQUENCY_STATUS",
            "LOAN_AGE",
            "REMAINING_MONTHS_TO_LEGAL_MATURITY",
            "MODIFICATION_FLAG",
            "ZERO_BALANCE_CODE",
            "ZERO_BALANCE_EFFECTIVE_DATE",
            "CURRENT_INTEREST_RATE",
            "CURRENT_NON_INTEREST_BEARING_UPB",
            "DUE_DATE_OF_LAST_PAID_INSTALLMENT",
            "INTEREST_RATE_STEP_INDICATOR",
            "ESTIMATED_LTV",
            "DELINQUENCY_DUE_TO_DISASTER",
            "BORROWER_ASSISTANCE_STATUS_CODE",
            "INTEREST_BEARING_UPB",
        ]

        self.ORIGINATION_ELIGIBLE = [
    "CREDIT_SCORE",
    "FIRST_TIME_HOMEBUYER_FLAG",
    "MSA",
    "MI_PERCENTAGE",
    "NUMBER_OF_UNITS",
    "OCCUPANCY_STATUS",
    "OCLTV",
    "DTI",
    "ORIGINAL_UPB",
    "LTV",
    "ORIGINAL_INTEREST_RATE",
    "CHANNEL",
    "PPM_FLAG",
    "PRODUCT_TYPE",
    "STATE",
    "PROPERTY_TYPE",
    "POSTAL_CODE",
    "LOAN_PURPOSE",
    "ORIGINAL_LOAN_TERM",
    "NUMBER_OF_BORROWERS",
    "SELLER_NAME",
    "SERVICER_NAME",
    "SUPER_CONFORMING_FLAG",
    "PRE_RELIEF_REFI_LOAN_SEQ",
    "PROGRAM_INDICATOR",
    "RELIEF_REFINANCE_INDICATOR",
    "PROPERTY_VALUATION_METHOD",
    "IO_FLAG",
    "MORTGAGE_INSURANCE_CANCELLATION",
]

        self.mode = mode

        self.orig = None
        self.hist = None
        self.loan_number = None

        if self.hist is not None and self.orig is not None:
           self.hist = hist
           self.orig = orig

        else:

            orig_file_columns = ["SOURCE_ROW_ID", *ORIGINATION_COLUMNS, "SOURCE_QUARTER"]
            performance_file_columns = ["SOURCE_ROW_ID", *PERFORMANCE_COLUMNS, "SOURCE_QUARTER"]

            self.orig = pd.read_csv(orig_path, header=None, sep='|', names=orig_file_columns)
            self.hist = pd.read_csv(hist_path, header=None, sep='|', names=performance_file_columns)

            self.orig = self.orig.drop(columns=["SOURCE_ROW_ID", "SOURCE_QUARTER"])
            self.hist = self.hist.drop(columns=["SOURCE_ROW_ID", "SOURCE_QUARTER"])
            self.loan_number = loan_number




        self.x_scaled = None
        self.y_scaled = None






    def apply(self):

        self.orig = self.orig[self.orig['LOAN_SEQUENCE_NUMBER'] == self.loan_number]
        self.hist = self.hist[self.hist['LOAN_SEQUENCE_NUMBER'] == self.loan_number]

        return self.clean()  # hist, orig -> non_scaled





    # ------------------------------------------------------------------
    # Historique
    # ------------------------------------------------------------------


    def clean(self ):



        def join_orig_hist(orig, hist,max_drop_ratio=0.01):
            logger = logging.getLogger(__name__)
            orig_enrichment_cols = [
                "LOAN_SEQUENCE_NUMBER",
                "ORIGINAL_LOAN_TERM",
                "ORIGINAL_UPB",
                "LTV",
            ]
            hist = hist.merge(
                orig[orig_enrichment_cols],
                on="LOAN_SEQUENCE_NUMBER",
                how="left",
            )

            # --- Bloc 1 : REMAINING_MONTHS_TO_LEGAL_MATURITY ---
            mask = hist["REMAINING_MONTHS_TO_LEGAL_MATURITY"].isna()
            if mask.any():
                hist.loc[mask, "REMAINING_MONTHS_TO_LEGAL_MATURITY"] = (
                        hist.loc[mask, "ORIGINAL_LOAN_TERM"]
                        - hist.loc[mask, "LOAN_AGE"]
                )

                hist, maturity_stats = self.handle_remaining_maturity_nan(
                    hist,
                    max_drop_ratio=max_drop_ratio,
                    logger=logger,
                    return_stats=True,
                )

            # --- Bloc 2 : ESTIMATED_LTV ---
            estimated_ltv_missing = hist['ESTIMATED_LTV'].isna()
            if estimated_ltv_missing.any():
                target_loans = hist.loc[
                    estimated_ltv_missing,
                    'LOAN_SEQUENCE_NUMBER'
                ].unique()

                loans_no_ltv = orig[
                    orig['LOAN_SEQUENCE_NUMBER'].isin(target_loans) & orig['LTV'].isna()
                    ]['LOAN_SEQUENCE_NUMBER'].unique()

                if len(loans_no_ltv) > 0:
                    orig = orig[~orig['LOAN_SEQUENCE_NUMBER'].isin(loans_no_ltv)]
                    hist = hist[~hist['LOAN_SEQUENCE_NUMBER'].isin(loans_no_ltv)]
                    estimated_ltv_missing = hist['ESTIMATED_LTV'].isna()
                    target_loans = hist.loc[
                        estimated_ltv_missing,
                        'LOAN_SEQUENCE_NUMBER'
                    ].unique()

                    # Garde : dataset vide après exclusion
                    if hist.empty:
                        raise ValueError(
                            f"Dataset vide après exclusion de {len(loans_no_ltv)} prêt(s) "
                            f"sans LTV reconstituable. Vérifier orig."
                        )

                if len(target_loans) > 0:
                    mask = hist['ESTIMATED_LTV'].isna()
                    valeur_bien = hist.loc[mask, 'ORIGINAL_UPB'] / (hist.loc[mask, 'LTV'] / 100)
                    hist.loc[mask, 'ESTIMATED_LTV'] = (hist.loc[mask, 'CURRENT_ACTUAL_UPB'] / valeur_bien) * 100

            hist = hist.drop(
                columns=["ORIGINAL_LOAN_TERM", "ORIGINAL_UPB", "LTV"],
                errors="ignore",
            )

            return hist, orig


        with ThreadPoolExecutor(max_workers=2) as executor:
            future_hist = executor.submit(self._clean_hist, self.hist)
            future_orig = executor.submit(self._clean_orig, self.orig)

            hist = future_hist.result()
            orig = future_orig.result()

        hist, orig = join_orig_hist(orig, hist)

        # Le drop est appliqué uniquement en mode PD.
        # En mode LGD, hist est retourné complet — la sélection des features
        # est déléguée à LGDFeaturePipeline.
        if self.mode == "pd":
            to_drop = [col for col in hist.columns if col not in self.ELIGIBLE_COLUMNS]
            hist = hist.drop(columns=to_drop)

        hist = hist.loc[:, ~hist.columns.duplicated()]

        return hist, orig





    # ------------------------------------------------------------------
    # Origination
    # ------------------------------------------------------------------

    def _clean_orig(self, orig: pd.DataFrame) -> pd.DataFrame:

        orig['VINTAGE'] = orig['LOAN_SEQUENCE_NUMBER'].str[1:5]

        # fillna AVANT astype sinon NaN devient "nan" string
        orig['MSA'] = orig['MSA'].fillna('RURAL').astype(str)
        orig['CREDIT_SCORE'] = orig['CREDIT_SCORE'].replace(9999, np.nan)
        orig['FIRST_TIME_HOMEBUYER_FLAG'] = orig['FIRST_TIME_HOMEBUYER_FLAG'].astype(object)
        orig['FIRST_TIME_HOMEBUYER_FLAG'] = orig['FIRST_TIME_HOMEBUYER_FLAG'].replace(9, np.nan)
        orig['MI_PERCENTAGE']       = orig['MI_PERCENTAGE'].replace(999, np.nan)
        orig['NUMBER_OF_UNITS']     = orig['NUMBER_OF_UNITS'].replace(99, np.nan)
        orig['OCCUPANCY_STATUS']    = orig['OCCUPANCY_STATUS'].replace(9, np.nan)
        orig['OCCUPANCY_STATUS']    = orig['OCCUPANCY_STATUS'].astype(object)

        orig['OCLTV']   = orig['OCLTV'].replace(999, np.nan)
        orig['DTI']     = orig['DTI'].replace(999, np.nan)
        orig['LTV']     = orig['LTV'].replace(999, np.nan)
        orig['CHANNEL'] = orig['CHANNEL'].replace(9, np.nan)
        orig['PPM_FLAG'] = orig['PPM_FLAG'].astype(object)

        orig['PROPERTY_TYPE'] = orig['PROPERTY_TYPE'].replace(99, np.nan)
        orig['LOAN_PURPOSE'] = orig['LOAN_PURPOSE'].replace(9, np.nan)
        orig['NUMBER_OF_BORROWERS'] = orig['NUMBER_OF_BORROWERS'].replace(99, np.nan)
        orig['PROGRAM_INDICATOR'] = orig['PROGRAM_INDICATOR'].astype(object)
        orig['MORTGAGE_INSURANCE_CANCELLATION'] = orig['MORTGAGE_INSURANCE_CANCELLATION'].astype(object)

        # SUPER_CONFORMING_FLAG — vide = non super conforme
        orig['SUPER_CONFORMING_FLAG'] = orig['SUPER_CONFORMING_FLAG'].fillna('N')

        # RELIEF_REFINANCE_INDICATOR — vide = non applicable
        orig['RELIEF_REFINANCE_INDICATOR'] = orig['RELIEF_REFINANCE_INDICATOR'].fillna('N')

        # OCLTV — remplacer par LTV si manquant
        orig['OCLTV'] = orig['OCLTV'].fillna(orig['LTV'])

        # LTV — remplacer par OCLTV si manquant
        orig['LTV'] = orig['LTV'].fillna(orig['OCLTV'])

        # NUMBER_OF_BORROWERS — médiane = 1
        orig['NUMBER_OF_BORROWERS'] = orig['NUMBER_OF_BORROWERS'].fillna(1)

        # Indicatrices avant imputation
        orig['IS_MISSING_CREDIT_SCORE'] = orig['CREDIT_SCORE'].isna().astype(int)
        orig['IS_MISSING_DTI'] = orig['DTI'].isna().astype(int)

        # Imputation par médiane par segment (VINTAGE x PROPERTY_TYPE)
        orig['CREDIT_SCORE'] = orig.groupby(['VINTAGE', 'PROPERTY_TYPE'])['CREDIT_SCORE'] \
            .transform(lambda x: x.fillna(x.median()))

        orig['DTI'] = orig.groupby(['VINTAGE', 'PROPERTY_TYPE'])['DTI'] \
            .transform(lambda x: x.fillna(x.median()))

        orig = orig.drop(columns=['VINTAGE'])
        ORIGINATION_ELIGIBLE = self.ORIGINATION_ELIGIBLE + ['IS_MISSING_CREDIT_SCORE', 'IS_MISSING_DTI',
                                                       'LOAN_SEQUENCE_NUMBER']
        to_drop = [col for col in orig.columns if col not in ORIGINATION_ELIGIBLE]

        orig = orig.drop(to_drop, axis=1)

        return orig



    def handle_remaining_maturity_nan(self, hist, max_drop_ratio=0.01, logger=None, return_stats=False):
        """
        Traite les NaN persistants dans REMAINING_MONTHS_TO_LEGAL_MATURITY après imputation.

        Règles :
        - si aucun NaN ne reste : retour inchangé ;
        - les prêts clôturés CURRENT_ACTUAL_UPB == 0 sont conservés ;
        - les lignes non imputables à cause de ORIGINAL_LOAN_TERM ou LOAN_AGE manquant sont supprimées ;
        - si le ratio de suppression dépasse max_drop_ratio : exception ;
        - si toutes les lignes d'un prêt sont supprimées, le prêt disparaît naturellement du dataframe.
        """

        target_col = "REMAINING_MONTHS_TO_LEGAL_MATURITY"
        loan_id_col = "LOAN_SEQUENCE_NUMBER"
        orig_term_col = "ORIGINAL_LOAN_TERM"
        loan_age_col = "LOAN_AGE"
        upb_col = "CURRENT_ACTUAL_UPB"

        logger = logger or logging.getLogger(__name__)

        required_cols = {
            target_col,
            loan_id_col,
            orig_term_col,
            loan_age_col,
            upb_col,
        }

        missing_cols = required_cols - set(hist.columns)
        if missing_cols:
            raise ValueError(
                f"Colonnes manquantes pour diagnostiquer {target_col} : {sorted(missing_cols)}"
            )

        initial_rows = len(hist)

        nan_mask = hist[target_col].isna()

        stats = {
            "initial_rows": initial_rows,
            "remaining_nan_rows": int(nan_mask.sum()),
            "dropped_rows": 0,
            "dropped_loans": 0,
            "kept_closed_rows": 0,
            "final_rows": initial_rows,
        }

        if not nan_mask.any():
            logger.info(
                "%s : aucun NaN après imputation, aucune suppression.",
                target_col,
            )
            return (hist, stats) if return_stats else hist

        closed_mask = hist[upb_col].eq(0)

        missing_orig_term_mask = hist[orig_term_col].isna()
        missing_loan_age_mask = hist[loan_age_col].isna()

        kept_closed_mask = nan_mask & closed_mask

        droppable_mask = (
                nan_mask
                & ~closed_mask
                & (missing_orig_term_mask | missing_loan_age_mask)
        )

        unresolved_mask = nan_mask & ~closed_mask & ~droppable_mask

        if unresolved_mask.any():
            unresolved_count = int(unresolved_mask.sum())
            sample_loans = (
                hist.loc[unresolved_mask, loan_id_col]
                .dropna()
                .astype(str)
                .unique()[:10]
                .tolist()
            )

            raise ValueError(
                f"{target_col} : {unresolved_count} lignes restent NaN sans cause récupérable "
                f"(ni prêt clôturé, ni {orig_term_col}/{loan_age_col} manquant). "
                f"Exemples de prêts : {sample_loans}"
            )

        dropped_rows = int(droppable_mask.sum())
        drop_ratio = dropped_rows / initial_rows if initial_rows else 0

        if drop_ratio > max_drop_ratio:
            raise ValueError(
                f"{target_col} : suppression refusée. "
                f"{dropped_rows}/{initial_rows} lignes seraient supprimées "
                f"({drop_ratio:.2%}), seuil max={max_drop_ratio:.2%}."
            )

        loans_before = set(hist[loan_id_col].dropna().unique())

        cleaned_hist = hist.loc[~droppable_mask].copy()

        loans_after = set(cleaned_hist[loan_id_col].dropna().unique())
        dropped_loans = len(loans_before - loans_after)

        stats.update(
            {
                "dropped_rows": dropped_rows,
                "dropped_loans": dropped_loans,
                "kept_closed_rows": int(kept_closed_mask.sum()),
                "final_rows": len(cleaned_hist),
            }
        )

        logger.info(
            "%s : %s lignes supprimées, %s prêts supprimés, %s lignes clôturées conservées, "
            "%s lignes finales.",
            target_col,
            stats["dropped_rows"],
            stats["dropped_loans"],
            stats["kept_closed_rows"],
            stats["final_rows"],
        )

        return (cleaned_hist, stats) if return_stats else cleaned_hist







    def _clean_hist(self, hist: pd.DataFrame) -> pd.DataFrame:
        # Dates
        hist['MONTHLY_REPORTING_PERIOD'] = pd.to_datetime(
            hist['MONTHLY_REPORTING_PERIOD'], format='%Y%m', errors='coerce'
        )
        hist['ZERO_BALANCE_EFFECTIVE_DATE'] = pd.to_datetime(hist['ZERO_BALANCE_EFFECTIVE_DATE'], format='%Y%m',
                                                             errors='coerce')
        hist['DUE_DATE_OF_LAST_PAID_INSTALLMENT'] = pd.to_datetime(hist['DUE_DATE_OF_LAST_PAID_INSTALLMENT'],
                                                                   format='%Y%m', errors='coerce')

        # Flags et codes
        hist['MODIFICATION_FLAG'] = hist['MODIFICATION_FLAG'].astype(object)
        hist['ZERO_BALANCE_CODE'] = hist['ZERO_BALANCE_CODE'].fillna(0).astype(object)
        hist['ZERO_BALANCE_CODE'] = (
            hist['ZERO_BALANCE_CODE']
            .replace('RA', -1)
            .pipe(pd.to_numeric, errors='coerce')
            .astype('Int64')
        )

        hist['BORROWER_ASSISTANCE_STATUS_CODE'] = hist['BORROWER_ASSISTANCE_STATUS_CODE'].fillna('N').astype(object)
        hist['DELINQUENCY_DUE_TO_DISASTER'] = hist['DELINQUENCY_DUE_TO_DISASTER'].fillna('N').astype(object)

        # Valeurs aberrantes
        hist['ESTIMATED_LTV'] = hist['ESTIMATED_LTV'].replace(999, np.nan)

        hist['MODIFICATION_FLAG'] = hist['MODIFICATION_FLAG'].fillna('N')

        hist['INTEREST_RATE_STEP_INDICATOR'] = hist['INTEREST_RATE_STEP_INDICATOR'].fillna('N')

        if self.mode == "lgd":
            # CURRENT_LOAN_DELINQUENCY_STATUS — 'XX' = statut inconnu
            hist['CURRENT_LOAN_DELINQUENCY_STATUS'] = (
                hist['CURRENT_LOAN_DELINQUENCY_STATUS']
                .replace('XX', np.nan)
                .pipe(pd.to_numeric, errors='coerce')
            )

            # Proceeds et recoveries — peuvent contenir 'U' (unavailable)
            hist['NET_SALE_PROCEEDS'] = (
                hist['NET_SALE_PROCEEDS']
                .replace('U', np.nan)
                .pipe(pd.to_numeric, errors='coerce')
            )

            for col in ['MI_RECOVERIES', 'NON_MI_RECOVERIES', 'TOTAL_EXPENSES', 'ACTUAL_LOSS_CALCULATION']:
                hist[col] = pd.to_numeric(hist[col], errors='coerce')

            # Flag paiement différé
            hist['PAYMENT_DEFERRAL_FLAG'] = hist['PAYMENT_DEFERRAL_FLAG'].fillna('N').astype(object)


        return hist
