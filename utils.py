from   functools import reduce
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
import uuid

import config
import trade_processing as tp

from tensorflow import keras

BASE_FEE = 9.95

HEADER_COLS = [
    'description',
    'open_margin',
    'max_profit',
    'stock_price',
    'minutes_to_expiry'
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

def _get_column_name_list(shuffle=False):
    leg_order = list(range(1,config.TOTAL_LEGS + 1))
    if shuffle:
        np.random.shuffle(leg_order)
    return reduce(
        lambda x, i: x + LEG_COLUMNS_TEMPLATE.format(num=i),
        leg_order,
        LEG_COLUMNS_TEMPLATE.format(num=leg_order.pop(0))
    ).split()

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

def apply_func_to_dfs_in_tarball(tarball_path, func):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_list = extract_and_get_file_list(tarball_path, tmpdir)
        for f in file_list:
            fpath = os.path.join(tmpdir, f)
            df = func(pd.read_pickle(fpath))
            df.to_pickle(fpath)
        subprocess.check_call(
            ['tar', '-C', tmpdir, '-cf', tarball_path] + file_list)

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

def sort_trades_df_columns(df):
    # We don't know what order the data came in wrt columns, but we know the
    # order we want it in.
    return df[
        [c for c in HEADER_COLS if c in df.columns] +_get_column_name_list()]

def randomize_legs_columns(df):
    # We want to make the model does not care about the order in which the legs
    # are presented to it
    df.columns = [c for c in HEADER_COLS if c in df.columns] + \
                 _get_column_name_list(shuffle=True)

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

def build_examples(
    ticker,
    total_strategies,
    max_margin=None,
    winning_profit=0,
    total_trades=1*10**6,
    l_to_w_ratio=3,
    randomize_legs=False,
    verbose=False,
):
    def log(msg):
        if verbose:
            print(msg)

    log('Winning profit: {}'.format(winning_profit * 100))

    win_fraction = 1/(l_to_w_ratio + 1)
    required_min_wins = int((total_trades * win_fraction) / total_strategies)
    required_min_losses = int(
        (total_trades * (1 - win_fraction)) / total_strategies)

    log('Min wins: {}'.format(required_min_wins))
    log('Min losses: {}'.format(required_min_losses))

    strats_dfs = {'win': {}, 'lose': {}}

    # Collect the expiry paths
    exp_paths = []
    data_dir = os.path.join(config.ML_DATA_DIR, ticker)
    exps = (os.path.splitext(f)[0] for f in os.listdir(data_dir)
                                   if f.endswith('.tar'))
    for e in exps:
        exp_paths.append(load_spreads(ticker, e))

    enough_wins = False
    enough_losses = False
    loop = 1
    for df in spreads_tarballs_to_generator(exp_paths):
        log('{:*^30}'.format(loop))
        loop += 1

        if max_margin is not None:
            df = df[df.open_margin <= max_margin]
        if df.shape[0] == 0:
            continue

        log('inpecting {} trades of types {}'.format(
            df.shape[0], sorted(df.description.unique()))
        )

        # Randomize the leg order to make the model more robust
        if randomize_legs:
            randomize_legs_columns(df)

        winning_indices = df.max_profit >= winning_profit
        for desc in df.description.unique():
            try:
                # Only collect as many wins as we need, but keep on collecting
                # and sorting them for the worst wins (hardest to classify)
                # until we have enough
                if not enough_wins:
                    strats_dfs['win'][desc] = pd.concat((
                        strats_dfs['win'][desc],
                        df[(df.description == desc) & winning_indices]
                    )).sort_values(
                        by='max_profit', ascending=True)[:required_min_wins]

                # Ditto for losses, but this time note that we're sorting in the
                # opposite direction because we want the best losses
                if not enough_losses:
                    strats_dfs['lose'][desc] = pd.concat((
                        strats_dfs['lose'][desc],
                        df[(df.description == desc) & ~winning_indices]
                    )).sort_values(
                        by='max_profit', ascending=False)[:required_min_losses]

            except KeyError:
                strats_dfs['win'][desc] = df[(
                    df.description == desc) & winning_indices]
                strats_dfs['lose'][desc] = df[(
                    df.description == desc) & ~winning_indices]

        # Are we done yet?
        min_wins = min((d.shape[0] for d in strats_dfs['win'].values()))
        min_losses = min((d.shape[0] for d in strats_dfs['lose'].values()))
        strat_count = len(strats_dfs['win'])
        log(
            ('{:>6}: {}\n'
             '{:>6}: {:>8} ({:.1%})\n'
             '{:>6}: {:>8} ({:.1%})\n').format(
                'strats', strat_count,
                'wins',   min_wins,   min_wins/required_min_wins,
                'losses', min_losses, min_losses/required_min_losses)
        )
        enough_losses = min_losses >= required_min_losses
        enough_wins = min_wins >= required_min_wins
        if strat_count == total_strategies and enough_losses and enough_wins:
            break

    # Concat and save and return the location of the file
    examples_dir = os.path.join(data_dir, 'examples')
    if not os.path.exists(examples_dir):
        os.mkdir(examples_dir)
    fpath = os.path.join(examples_dir, str(uuid.uuid4()))
    pd.concat(
        [d for d in strats_dfs['win'].values()] +
        [d for d in strats_dfs['lose'].values()]
    ).to_pickle(fpath)

    return fpath

def calculate_fee(count=1, both_sides=True):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee / 100
