RAW_X_PATH = "data/X.csv"
RAW_Y_PATH = "data/target.csv"
PROCESSED_PATH = "data/processed.parquet"
PD_MODEL_PATH = "models/pd_model.joblib"
LGD_MODEL_PATH = "models/lgd_model.joblib"
METRICS_PATH = "models/metrics.json"
FIGURES_DIR = "app/static/figures"

RANDOM_STATE = 42
LOAN_TERM = 36
N_ITER = 12      # sampled candidates per model in RandomizedSearchCV
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Out-of-time split on issue date (YYYYMM). Only vintages whose 36m loans are fully resolved.
TRAIN_END = 201406
VALID_END = 201412
TEST_END  = 201512

# Net income of a performing loan as a share of the funded amount.
# Fully paid 36m loans (2007-2015) earned 15.5% gross interest on average;
# 0.10 is a net figure after servicing fees and funding cost.
# Approve if (1 - PD) * MARGIN > PD * LGD * CCF  ->  t* = MARGIN / (MARGIN + LGD * CCF)
MARGIN = 0.10
MARGIN_GRID = [0.05, 0.10, 0.155]

# Columns used to build the targets: never features
TARGET_SOURCE_COLS = [
    'princ_rec',
    'interest_rec',
    'late_fees_rec',
    'paym_rec_for_tot_amnt_fund',
    'remaining_princ_for_tot_amnt_fund'
]

NUM_FEATURES = [
    'funded_amnt',
    'interest_rate',
    'monthly_payment',
    'annual_income',
    'dept_paym_income_ratio',
    'emp_length_years',
    'num_30+_delinq_in_2yrs',
    'num_inq_in_6mths',
    'mths_since_last_delinq',
    'num_open_credit_lines',
    'num_derogatory_pub_rec',
    'total_credit_revolving_bal',
    'used_credit_share',
    'tot_num_credit_lines',
    'credit_history_years'
]

CAT_FEATURES = [
    'grade',
    'home_ownership_status',
    'verification_status',
    'loan_purpose',
    'addr_state'
]

FEATURES = NUM_FEATURES + CAT_FEATURES

# Subset known at application time: the deployed model and the /predict form
APP_NUM_FEATURES = [
    'funded_amnt',
    'interest_rate',
    'annual_income',
    'dept_paym_income_ratio',
    'emp_length_years',
    'num_30+_delinq_in_2yrs',
    'num_inq_in_6mths',
    'num_open_credit_lines',
    'total_credit_revolving_bal',
    'used_credit_share',
    'credit_history_years'
]

APP_CAT_FEATURES = [
    'grade',
    'home_ownership_status',
    'verification_status',
    'loan_purpose'
]

APP_FEATURES = APP_NUM_FEATURES + APP_CAT_FEATURES
TARGETS = ['default', 'lgd', 'ead_default', 'recoveries']
