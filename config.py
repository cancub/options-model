STORAGE_DIR     = 'pickles'
BACKUPS_DIR     = 'backups'
TRADES_DIR      = 'trades'

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
MARGIN          = 60
