STORAGE_DIR     = 'pickles'
BACKUPS_DIR     = 'backups'
META_COLUMNS    = ['symbolId', 'type', 'strike']
# DATA_NAMES      = ['lastTradePrice', 'volume', 'volatility', 'delta', 'gamma',
#                    'theta', 'vega', 'rho', 'openInterest']
DATA_NAMES      = ['bidPrice', 'askPrice', 'volume', 'volatility', 'delta', 'gamma',
                   'theta', 'vega', 'rho', 'openInterest']
MULTITHREADED   = True
QUIET           = True

'''
NOTE:
when reading the numpy files:
[datatype, timepoint, optionID]
'''
