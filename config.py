from   datetime import datetime, timezone
import os
import pytz

STORAGE_DIR     = os.path.abspath('./pickles')
BACKUPS_DIR     = os.path.abspath('./backups')
EXPIRIES_DIR    = os.path.abspath('./expiries')
TRADES_DIR      = os.path.abspath('./trades')
ML_DIR          = os.path.abspath('./learning')
ML_DATA_DIR     = os.path.join(ML_DIR, 'data')
ML_MODELS_DIR   = os.path.join(ML_DIR, 'models')

DATA_NAMES      = ['bidPrice', 'askPrice', 'volume', 'volatility', 'delta',
                    'gamma', 'theta', 'vega', 'rho', 'openInterest']
TICKERS         = ['FB', 'LYB', 'SPY', 'TSLA', 'QQQ', 'AAPL', 'EEM', 'IWM',
                   'NFLX', 'AMD', 'NVDA', 'GLD', 'SLV', 'VXX', 'GE', 'BAC',
                   'BA', 'AAL', 'NAT', 'CCL', 'F', 'DIS', 'BABA', 'PFE' ,'SBUX',
                   'M', 'ZM', 'SNAP', 'AMZN', 'C', 'ROKU', 'BBBY', 'UBER',
                   'SDC', 'INTC', 'SPCE', 'TWTR', 'TLRY', 'MSFT']
QUIET           = True

# Maximum amount of margin we are allowed to use for trades
# NOTE: this is /100 because of the scale of options pricing
MARGIN          = 20

# We're not going to be using this data for training in any case
IGNORE_LOSS     = 0

TOTAL_LEGS      = 4

BASE_FEE        = 9.95

HEADER_COLS     = [
    'description',
    'expiry',
    'expiry_timestamp',
    'open_time',
    'seconds_to_expiry',
    'stock_price',
    'open_margin',
    'max_profit',
]

LEG_COLUMNS_TEMPLATE = '''
    leg{num}_type
    leg{num}_strike
    leg{num}_credit
    leg{num}_volume
    leg{num}_volatility
    leg{num}_delta
    leg{num}_gamma
    leg{num}_theta
    leg{num}_vega
    leg{num}_rho
    leg{num}_openInterest
'''

EPOCH = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

DST_TRANSITIONS = pytz.timezone('America/Toronto')._utc_transition_times
