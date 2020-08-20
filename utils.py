import datetime as dt
from   io import BytesIO
import numpy as np
import os
import pandas as pd
import re
import subprocess as sp

import config
import trade_processing as tp

BASE_FEE = 9.95

def get_basic_datetimes(times_strings):
    dt_match = re.compile(r'\d+-\d+-\d+ \d+:\d+')
    strptime_format = '%Y-%m-%d %H:%M'
    return np.array(list(map(
        lambda x: dt.datetime.strptime(
            re.match(dt_match, x).group(0), strptime_format),
        times_strings
    )))

def get_last_backup_path():
    return os.path.join(
        config.BACKUPS_DIR,
        sorted(
            (b for b in os.listdir(config.BACKUPS_DIR) if b.endswith('.tar'))
        )[-1]
    )

def retrieve_options(ticker, expiry=None):
    # Determine the last backup file
    last_backup = get_last_backup_path()

    # Retrieve the pickle from the backup
    df = pd.read_pickle(
        BytesIO(sp.check_output(['tar', '-xOf', last_backup, ticker + '.bz2'])),
        compression='bz2'
    )

    # Select the expiry if specified
    if expiry is not None:
        df = df.xs(expiry, level=1)

    return df

def load_spreads(ticker, expiry):
    df_path = os.path.join(
        config.TRADES_DIR, '{}_{}_spreads'.format(ticker, expiry))

    # First see if we've already built these spreads
    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        # Nope, so load them
        df = tp.collect_spreads(ticker=ticker, expiry=expiry)
        # And then save them for next time
        df.to_pickle(df_path)

    return df

def calculate_fee(count, both_sides=False):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee
