from   io import BytesIO
import json
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import subprocess
import tarfile
import tempfile

import config
import trade_processing as tp

from tensorflow import keras

BASE_FEE = 9.95

def get_last_backup_path():
    fnames = (b for b in os.listdir(config.BACKUPS_DIR) if b.endswith('.tar'))
    return os.path.abspath(os.path.join(config.BACKUPS_DIR, sorted(fnames)[-1]))

def load_options(ticker, expiry=None):
    # First try to load one of the expiry files
    try:
        df = pd.read_pickle(
            os.path.join(config.EXPIRIES_DIR, ticker, '{}.bz2'.format(expiry)))
    except FileNotFoundError:
        # Nope, not there (probably looking ahead to future expiries). So now we
        # need to open up the last backup file
        last_backup = get_last_backup_path()

        # Retrieve the pickle from the backup
        df = pd.read_pickle(
            BytesIO(
                subprocess.check_output(
                    ['tar', '-xOf', last_backup, ticker + '.bz2'])
            ),
            compression='bz2'
        )

    df = df.xs(expiry, level=1)

    return df

def load_spreads(
    ticker,
    expiry,
    winning_profit=None,
    loss_win_ratio=None,
    refresh=False,
    verbose=False
):
    ticker_dir = os.path.join(config.ML_DATA_DIR, ticker)
    spreads_path = os.path.join(ticker_dir, '{}.tar'.format(expiry))

    # Do we want to load the spreads from scratch?
    if not refresh:
        if verbose:
            print('Attempting to locate saved spreads')
        if os.path.exists(spreads_path):
            if verbose:
                print('Saved spreads located.')
            return spreads_path
        if verbose:
            print('No spreads saved.')

    if verbose:
        print('Loading options')

    options_df = load_options(ticker, expiry)

    if verbose:
        print('Building spreads.')

    out_dir = tp.collect_spreads(ticker,
                                 expiry,
                                 options_df,
                                 winning_profit=winning_profit,
                                 loss_win_ratio=loss_win_ratio,
                                 verbose=verbose)

    # Save these so that we don't have to reload them next time
    if verbose:
        print('Adding spreads to datastore.')

    if not os.path.exists(ticker_dir):
        if verbose:
            print('Creating ticker directory {}.'.format(ticker_dir))
        os.makedirs(ticker_dir)

    # Package it into a tarball
    subprocess.check_call(
        ['tar', '-C', out_dir, '-cf', spreads_path] + os.listdir(out_dir))

    shutil.rmtree(out_dir)

    return spreads_path

def extract_and_get_file_list(tarball_path, output_dir):
    files_bytes = subprocess.check_output(
        ['tar', '-C', output_dir, '-xvf', tarball_path])
    return sorted(files_bytes.decode('utf-8').strip().split('\n'))

def spreads_tarballs_to_generator(tarball_paths, shuffle=True):
    # First we need to get a list of all of the files to be loaded
    if not isinstance(tarball_paths, list):
        tarball_paths = [tarball_paths]
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for p in tarball_paths:
            file_list = extract_and_get_file_list(p, tmpdir)
            paths += [os.path.join(tmpdir, f) for f in file_list]
        if shuffle:
            random.shuffle(paths)
        for p in paths:
            yield sort_trades_df_columns(pd.read_pickle(p))

def load_best_model(ticker, max_margin=np.inf, min_profit = 0):
    # Find the model related to these values which has the lowest loss
    model_dir = os.path.join(config.ML_MODELS_DIR, ticker)
    best_model_tarball = None
    lowest_loss = np.inf
    for fname in (f for f in os.listdir(model_dir) if f.endswith('.tar')):
        fpath = os.path.join(model_dir, fname)

        # Get the metadata
        meta = json.loads(
            subprocess.check_output(['tar', '-xOf', fpath, 'metadata']))

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
        stds = pd.read_pickle(os.path.join(tmpdir, 'variances')).pow(1/2)

        with open(os.path.join(tmpdir, 'metadata'), 'r') as MF:
            metadata = json.load(MF)

    return {
        'model': model,
        'means': means,
        'stds': stds,
        'metadata': metadata
    }

def get_predictions(
    viable_spreads,
    options_model=None,
    ticker=None,
    max_margin=np.inf,
    min_profit=0
):

    if options_model is None:
        options_model = load_best_model(ticker, max_margin=np.inf, min_profit=0)

    model = options_model['model']
    means = options_model['means']
    stds = options_model['stds']
    columns_order = options_model['metadata']['feature_order']

    # We need to compile to continue
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True))

    # Maybe we got the margin or profits in with the spreads. These were not
    # provided to the model and so they must be removed
    examples = viable_spreads[columns_order]

    # Make sure we have the right columns in the right order
    means = means[columns_order]
    stds = stds[columns_order]

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
    for i in [1, 2, 3, 4]:
        if 'leg{}_type'.format(i) not in trades_df.columns:
            break
        meta_cols += leg_col_names.format(num=i).split()

    return trades_df[meta_cols]

def set_standard_static_stats(means, variances):
    # Some columns should not be normalized
    columns = means.index.values
    static_columns = ['open_margin', 'max_profit']
    for i in [1, 2, 3, 4]:
        type_col = 'leg{}_type'.format(i)
        if type_col in columns:
            static_columns.append(type_col)

    for c in static_columns:
        try:
            means[c] = 0
            variances[c] = 1
        except KeyError:
            pass

    return means, variances

def collect_statistics(trades_df):
    trades_means = trades_df.mean()
    trades_vars = trades_df.var()

    return set_standard_static_stats(trades_means, trades_vars)

def pool_stats_from_stats_df(ticker):
    stats_df = pd.read_pickle(os.path.join(
        config.ML_DATA_DIR, ticker, 'stats'))

    pools = stats_df.pop('pools')
    samples = stats_df.pop('samples')

    means = stats_df.xs('mean', level=1)
    variances = stats_df.xs('variance', level=1)

    new_means = means.multiply(samples, 0).sum() / samples.sum()
    new_variances = variances.multiply(
        samples-pools, 0).sum() / (samples.sum() - pools.sum())

    # Ensure that some columns will not be normalized
    return set_standard_static_stats(new_means, new_variances)

def pool_stats_from_expiry(expiry_path):

    # Collect the statistics
    means = []
    variances = []
    sample_sizes = []

    # Get the stats for each DataFrame in the expiry tarball
    for df in spreads_tarballs_to_generator(expiry_path, shuffle=True):
        df_means, df_vars = collect_statistics(df)
        means.append(df_means)
        variances.append(df_vars)
        sample_sizes.append(df.shape[0])

    # Collect the numerators for both means and variances
    pooled_means = None
    pooled_vars = None
    total_samples = sum(sample_sizes)
    for i in range(len(sample_sizes)):
        next_mean = means[i] * sample_sizes[i]
        next_var = variances[i] * (sample_sizes[i] - 1)
        try:
            pooled_means += next_mean
            pooled_vars += next_var
        except TypeError:
            pooled_means = next_mean
            pooled_vars = next_var

    # Divide the result by the respective denominators
    pool_count = len(sample_sizes)
    pooled_means /= total_samples
    pooled_vars /= (total_samples - pool_count)

    # Finally, reset some of the values that should not be changed
    pooled_means, pooled_vars = set_standard_static_stats(pooled_means,
                                                          pooled_vars)

    # Also provide the number of samples with the stats for combining these
    # pooled values with other pooled values later on
    return pooled_means, pooled_vars, total_samples, pool_count

def normalize_metadata_columns(trades_df):
    # We must not normalize the leg types since these columns are categorical.
    # So we give these specific columns mean 0 std 1 to make them unchanged
    # after the normalization operation
    meta_means, meta_vars = collect_statistics(trades_df)

    meta_stds = meta_vars.pow(1/2)

    normalized_df = (trades_df - meta_means) / meta_stds

    return normalized_df, meta_means, meta_stds

def collect_winners_and_hard_losers(trades_df,
                                    winning_profit=0,
                                    l_to_w_ratio=3):

    def get_and_print_wl(df):
        total_trades = len(df)

        # Determine the max profits when purchasing one of these trades
        profit_less_fees = df.max_profit - calculate_fee()

        losers = df[profit_less_fees < winning_profit]
        winners = df[profit_less_fees >= winning_profit]
        total_winners = len(winners)
        total_losers = len(losers)

        print(
            '{} ({:.1%}) winners\n{} ({:.1%}) losers\n'.format(
                total_winners, total_winners / total_trades,
                total_losers, total_losers / total_trades
            )
        )

        return winners, losers

    print('Before')
    winners, losers = get_and_print_wl(trades_df)

    # Get at most the desired ratio of losers to winners, using the losers that
    # were closest to profit
    losers_to_get = winners.shape[0] * l_to_w_ratio

    return_df = pd.concat((
        winners,
        losers.sort_values(by='max_profit', ascending=False)[:losers_to_get]
    ))

    print('After')
    get_and_print_wl(return_df)
    return return_df

def calculate_fee(count=1, both_sides=True):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee / 100
