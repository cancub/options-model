from   datetime import datetime, timezone
import os
import pytz

STORAGE_DIR     = os.path.abspath('./pickles')
BACKUPS_DIR     = os.path.abspath('./backups')
SPREADS_DIR     = os.path.abspath('./spreads')
EXPIRIES_DIR    = os.path.abspath('./expiries')
ML_DIR          = os.path.abspath('./learning')
ML_DATA_DIR     = os.path.join(ML_DIR, 'data')
ML_MODELS_DIR   = os.path.join(ML_DIR, 'models')

DATA_NAMES = [
    'symbolId',
    'bidPrice',
    'askPrice',
    'volume',
    'volatility',
    'delta',
    'gamma',
    'theta',
    'vega',
    'rho',
    'openInterest'
]

TICKERS = [
    'AAL',
    'AAPL',
    'AMD',
    'AMZN',
    'BA',
    'BABA',
    'BAC',
    'BBBY',
    'C',
    'CCL',
    'DIS',
    'EEM',
    'F',
    'FB',
    'GE',
    'GLD',
    'INTC',
    'IWM',
    'LYB',
    'M',
    'MSFT',
    'NAT',
    'NFLX',
    'NVDA',
    'PFE',
    'QQQ',
    'ROKU',
    'SBUX',
    'SDC',
    'SLV',
    'SNAP',
    'SPCE',
    'SPY',
    'TLRY',
    'TSLA',
    'TWTR',
    'UBER',
    'VXX',
    'ZM',
]

QUIET           = True

# Maximum amount of margin we are allowed to use for trades
# NOTE: this is /100 because of the scale of options pricing
MARGIN          = 20

# We're not going to be using this data for training in any case
IGNORE_LOSS     = 0

TOTAL_LEGS      = 4

# 9.95 USD -> CAD
BASE_FEE        = 13

COLUMN_ORDER    = [
    'description',
    'expiry',
    'expiry_dow',
    'expiry_wom',
    'open_time',
    'seconds_to_expiry',
    'stock_price',
    'open_margin',
    'max_profit',
    'option_type',
    'option_type_cat',

    'leg1_symbolId',
    'leg1_strike',
    'leg1_action',
    'leg1_credit',
    'leg1_volume',
    'leg1_volatility',
    'leg1_delta',
    'leg1_gamma',
    'leg1_theta',
    'leg1_vega',
    'leg1_rho',
    'leg1_openInterest',

    'leg2_symbolId',
    'leg2_strike',
    'leg2_action',
    'leg2_credit',
    'leg2_volume',
    'leg2_volatility',
    'leg2_delta',
    'leg2_gamma',
    'leg2_theta',
    'leg2_vega',
    'leg2_rho',
    'leg2_openInterest',

    'leg3_symbolId',
    'leg3_strike',
    'leg3_action',
    'leg3_credit',
    'leg3_volume',
    'leg3_volatility',
    'leg3_delta',
    'leg3_gamma',
    'leg3_theta',
    'leg3_vega',
    'leg3_rho',
    'leg3_openInterest',

    'leg4_symbolId',
    'leg4_strike',
    'leg4_action',
    'leg4_credit',
    'leg4_volume',
    'leg4_volatility',
    'leg4_delta',
    'leg4_gamma',
    'leg4_theta',
    'leg4_vega',
    'leg4_rho',
    'leg4_openInterest',
]

# The columns we will not use for prediction and therefor must be excluded from
# any statistics.
STATS_IGNORE_COLS = [
    'description',
    'expiry',
    'open_time',
    'open_margin',
    'max_profit',
    'option_type',

    'leg1_symbolId',
    'leg2_symbolId',
    'leg3_symbolId',
    'leg4_symbolId',
]

# The columns which will eventually be used to train a model
STATS_COLS = [c for c in COLUMN_ORDER if c not in STATS_IGNORE_COLS]

# The columns which must be min-max normalized.
MIN_MAX_NORM_COLS = [
    'expiry_dow',
    'expiry_wom',
]

LOG_NORM_COLS = []

# The columns which should be left alone with regard to normalization. Likely
# because they are already normalized by another means or they represent a
# category value that can take on a neutral value.
NON_NORM_COLS = [
    'option_type_cat',
    'leg1_action',
    'leg2_action',
    'leg3_action',
    'leg4_action',
] + MIN_MAX_NORM_COLS + LOG_NORM_COLS

# The remainder fo the columns will be standardized via mean and std.
STANDARDIZE_COLS = [
    c for c in STATS_COLS
        if c not in NON_NORM_COLS + STATS_IGNORE_COLS
]

EPOCH = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

DST_TRANSITIONS = pytz.timezone('America/Toronto')._utc_transition_times
