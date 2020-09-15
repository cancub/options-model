import datetime as dt
from   io import BytesIO
import json
import numpy as np
import os
import pandas as pd
import re
import subprocess as sp
import tarfile
import tempfile

import config
import trade_processing as tp

from tensorflow import keras

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
    fnames = (b for b in os.listdir(config.BACKUPS_DIR) if b.endswith('.tar'))
    return os.path.abspath(os.path.join(config.BACKUPS_DIR, sorted(fnames)[-1]))

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

def load_spreads(ticker, expiry, refresh=False, save=True, verbose=False):
    filepath = os.path.join(
        config.ML_DATA_DIR, '{}_{}_spreads.bz2'.format(ticker, expiry))
    if not refresh:
        if verbose:
            print('Attempting to load saved spreads')
        try:
            df = pd.read_pickle(filepath)
            print('Loaded')
            return df
        except FileNotFoundError:
            if verbose:
                print('No spreads saved.')

    if verbose:
        print('Building spreads.')
    df = tp.collect_spreads(
            ticker, expiry, get_max_profit=True, verbose=verbose)
    if save:
        # Save these so that we don't have to reload them next time
        if verbose:
            print('Saving spreads to file: {}'.format(filepath))
        df.to_pickle(filepath)

    return df

def get_predictions(ticker, viable_spreads, max_margin=np.inf, min_profit = 0):
    viable_spreads = sort_trades_df_columns(viable_spreads)

    # Find the model related to these values which has the lowest loss
    model_dir = os.path.join(config.ML_MODELS_DIR, ticker)
    best_model_tarball = None
    lowest_loss = np.inf
    for fname in (f for f in os.listdir(model_dir) if f.endswith('.tar')):
        fpath = os.path.join(model_dir, fname)

        # Get the metadata
        meta = json.loads(sp.check_output(['tar', '-xOf', fpath, 'metadata']))

        # Check the criteria
        if (meta['max_margin'] < max_margin
                or meta['min_profit'] != min_profit
                or meta['loss'] >= lowest_loss):
            continue
        lowest_loss = meta['loss']
        best_model_tarball = fpath

    # Load the model and statistics from the tarball
    model_tarball = tarfile.open(best_model_tarball)
    with tempfile.TemporaryDirectory() as tmpdir:

        # Extract the model into a temporary directory
        model_tarball.extractall(tmpdir)

        # Load the model
        model = keras.models.load_model(
            os.path.join(tmpdir, 'checkpoint'), compile=False)

        # Load the stats
        means = pd.read_pickle(os.path.join(tmpdir, 'means'))
        stds = pd.read_pickle(os.path.join(tmpdir, 'stds'))

    # We need to compile to continue
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True))

    # Maybe we got the margin or profits in with the spreads. These were not
    # provided to the model and so they must be removed
    examples = viable_spreads
    for col in ('open_margin', 'max_profit'):
        if col in examples.columns:
            examples = examples.drop(col, axis=1)

    # Make sure we have the right columns in the right order
    means = means[examples.columns]
    stds = stds[examples.columns]

    # Normalize
    examples = (examples - means) / stds

    # Get the predictions
    results = model.predict(examples.values)
    viable_spreads.insert(0, 'confidence', results)

    return viable_spreads.sort_values(by=['confidence'], ascending=False)

def sort_trades_df_columns(trades_df):
    # We don't know what order the data came in wrt columns, but we know the
    # order we want it in
    meta_cols = []
    header_cols = [
        'open_margin',
        'max_profit',
        'stock_price',
        'minutes_to_expiry'
    ]
    for col in (c for c in header_cols if c in trades_df.columns):
        meta_cols.append(col)
    leg_col_names = '''
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
    for i in range(1,5):
        if 'leg{}_type'.format(i) not in trades_df.columns:
            break
        meta_cols += leg_col_names.format(num=i).split()

    return trades_df[meta_cols]

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

def collect_winners_and_hard_losers(trades_df,
                                    winning_profit=0,
                                    l_to_w_ratio=3):

    total_trades = len(trades_df)

    # Determine the max profits when purchasing one of these trades
    profit_less_fees = trades_df.max_profit - calculate_fee(1, both_sides=True)

    losers = trades_df[profit_less_fees < winning_profit]
    winners = trades_df[profit_less_fees >= 0]
    total_winners = len(winners)
    total_losers = len(losers)

    print(
        '{} ({:.1%}) winners\n{} ({:.1%}) losers'.format(
            total_winners, total_winners / total_trades,
            total_losers, total_losers / total_trades
        )
    )

    # Get at most the desired ratio of losers to winners, using the losers that
    # were closest to profit
    losers_to_get = total_winners * l_to_w_ratio

    return pd.concat((
        winners,
        losers.sort_values(by='max_profit', ascending=False)[:losers_to_get]
    ))

def calculate_fee(count=1, both_sides=True):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee
