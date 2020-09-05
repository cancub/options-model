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

def load_options(ticker, expiry=None):
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

def load_spreads(ticker, expiry, verbose=False):
    filepath = os.path.join(
        config.ML_DATA_DIR, '{}_{}_spreads.bz2'.format(ticker, expiry))
    if verbose:
        print('Attempting to load saved spreads')
    try:
        df = pd.read_pickle(filepath)
        print('Loaded')
        return df
    except FileNotFoundError:
        if verbose:
            print('No spreads saved. Building spreads')
        df = sort_trades_df_columns(
            tp.collect_spreads(ticker, expiry, verbose=verbose))
        # Save these so that we don't have to reload them next time
        if verbose:
            print('Saving spreads to file: {}'.format(filepath))
        df.to_pickle(filepath)

    return df

def normalize_metadata_columns(trades_df):
    # We must not normalize the leg types since these columns are categorical.
    # So we give these specific columns mean 0 std 1 to make them unchanged
    # after the normalization operation
    meta_means = trades_df.mean()
    meta_stds = trades_df.std()

    for i in range(1, 5):
        type_str = 'leg{}_type'.format(i)
        if type_str not in trades_df.columns:
            break
        meta_means[type_str] = 0
        meta_stds[type_str] = 1

    # Also ignore the selection data
    meta_means.open_margin = 0
    meta_means.max_profit = 0
    meta_stds.open_margin = 1
    meta_stds.max_profit = 1

    normalized_df = (trades_df - meta_means) / meta_stds

    return normalized_df, meta_means, meta_stds

def calculate_fee(count, both_sides=False):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee
