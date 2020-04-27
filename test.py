from datetime import datetime
import numpy as np
import os
import pandas as pd
import shutil
import threading

import config
from data_management import update_data_files

lock = threading.Lock()

ex = '2030-09-15'
ticker = 'TEST'

# META_COLUMNS = ['symbolId', 'type', 'strike']
# DATA_NAMES = ['lastTradePrice', 'volume', 'volatility', 'delta', 'gamma',
#               'theta', 'vega', 'rho', 'openInterest']

series_data = {}
for i in range(1,4):
    sId = int('{0}{0}{0}'.format(i))
    series_data[sId] = {'type': 'C', 'strike': i, 'data':{}}
    for j in range(len(config.DATA_NAMES)):
        series_data[sId]['data'][config.DATA_NAMES[j]] = j*i

update_data_files(ticker, ex, series_data, datetime.now(), lock)

base_dir = os.path.join('pickles',ex)
base_path = os.path.join(base_dir, ticker)

# Now load in the files we just created 
data_arr = np.load(base_path)[:, :, :-1]
meta_df = pd.read_pickle(base_path + '_meta')[:-1]

# Remove the second element from each
with open(base_path, 'wb') as NF:
    np.save(NF, data_arr)
meta_df.to_pickle(base_path + '_meta')


# Now when we run the update, we should get a column of NaN for just the first
# timepoint
update_data_files(ticker, ex, series_data, datetime.now(), lock)

# Now load in the files we just created
data_arr2 = np.load(base_path)
meta_df2 = pd.read_pickle(base_path + '_meta')

# And confirm that the values are what we expect
for i in range(1, 4):
    for j in range(len(config.DATA_NAMES)):
        for k in range(2):
            if i == 3 and k == 0:
                # We deleted these values for the first datapoint
                assert(np.isnan(data_arr2[j, k, i-1]))
            else:
                assert(data_arr2[j, k, i-1] == j*i)

shutil.rmtree(base_dir)


