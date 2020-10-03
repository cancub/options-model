from   datetime import datetime, timedelta, timezone
from   functools import reduce
from   io import BytesIO
import json
import numpy as np
import os
import pandas as pd
import pytz
import random
import re
import shutil
import subprocess
import tempfile
import uuid

import config
import trade_processing as tp

from questrade_helpers import QuestradeSecurities

def _get_column_name_list(shuffle=False):
    leg_order = list(range(1,config.TOTAL_LEGS + 1))
    if shuffle:
        np.random.shuffle(leg_order)
    return reduce(
        lambda x, i: x + config.LEG_COLUMNS_TEMPLATE.format(num=i),
        leg_order,
        config.LEG_COLUMNS_TEMPLATE.format(num=leg_order.pop(0))
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
    with tempfile.TemporaryDirectory(prefix='options_func') as tmpdir:
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

def spreads_tarballs_to_generator(tarball_paths, count=None, shuffle=True):
    # First we need to get a list of all of the files to be loaded
    if not isinstance(tarball_paths, list):
        tarball_paths = [tarball_paths]
    with tempfile.TemporaryDirectory(prefix='spreads_gen') as tmpdir:
        paths = []
        for p in tarball_paths:
            file_list = extract_and_get_file_list(p, tmpdir)
            paths += [os.path.join(tmpdir, f) for f in file_list]
        if shuffle:
            random.shuffle(paths)
        for p in paths:
            yield sort_trades_df_columns(pd.read_pickle(p))
            if count != None:
                count -= 1
                if count == 0:
                    break

def ticker_expiry_to_generator(ticker, expiry, count=None, shuffle=True):
    # First we need to get a list of all of the files to be loaded
    return spreads_tarballs_to_generator(load_spreads(ticker, expiry),
                                         count,
                                         shuffle)

def sort_trades_df_columns(df):
    # We don't know what order the data came in wrt columns, but we know the
    # order we want it in.
    return df[
        [c for c in config.HEADER_COLS if c in df.columns] +
        _get_column_name_list()
    ]

def randomize_legs_columns(df):
    # We want to make the model does not care about the order in which the legs
    # are presented to it
    df.columns = [c for c in config.HEADER_COLS if c in df.columns] + \
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

def build_examples(
    ticker,
    max_margin=None,
    winning_profit=0,
    total_trades=1*10**6,
    l_to_w_ratio=1,
    randomize_legs=False,
    hard_winners=False,
    win_pool_multiplier=1,
    hard_losers=True,
    loss_pool_multiplier=1,
    save_dir=None,
    verbose=False,
):

    def log(msg):
        if verbose:
            print(msg)

    def strats_paths_to_generators(strats_paths):

        def strat_paths_to_generator(strat_paths):
            np.random.shuffle(strat_paths)
            for p in strat_paths:
                yield pd.read_pickle(p)

        def path_to_name(p):
            return os.path.split(p)[1].split('-')[0]

        strats_dicts = {}

        # Walk through the files and append them to their respective list of
        # strategy DataFrame paths in the dictionary
        for p in strats_paths:
            try:
                strats_dicts[path_to_name(p)].append(p)
            except KeyError:
                strats_dicts[path_to_name(p)] = [p]

        return {k: strat_paths_to_generator(v) for k, v in strats_dicts.items()}

    strategy_generators = {}
    strats_dfs          = {'win': {}, 'loss': {}}
    enough_wins         = False
    enough_losses       = False

    log('Winning profit: {}'.format(winning_profit * 100))

    # Get the paths to the available spreads tarballs
    data_dir = os.path.join(config.ML_DATA_DIR, ticker)
    tarball_paths = (os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if f.endswith('.tar'))

    # Extract all of the available strategy DataFrames for all available
    # expiries into one directory
    with tempfile.TemporaryDirectory(prefix='options_examples') as tmpdir:
        file_list = []
        for p in tarball_paths:
            # Extract it and get the file list
            file_list += list(map(
                lambda f: os.path.join(tmpdir, f),
                extract_and_get_file_list(p, tmpdir)
            ))

        # Convert each list of strategies into a generator of DataFrames for
        # that strategy
        strategy_generators = strats_paths_to_generators(file_list)

        strat_names = list(strategy_generators.keys())

        # Now that we know the names of the unique strategies, we can count how
        # many there are ...
        total_strategies = len(strat_names)

        # ... determine how many winners and losers are needed for each
        # strategy ...
        win_frac = 1/(l_to_w_ratio + 1)
        min_wins = int((total_trades * win_frac) / total_strategies)
        min_losses = int((total_trades * (1 - win_frac)) / total_strategies)

        win_pool_size = min_wins * \
            (1 if not hard_winners else win_pool_multiplier)
        loss_pool_size = min_losses * \
            (1 if not hard_losers else loss_pool_multiplier)

        log('Min wins: {}'.format(min_wins))
        log('Min losses: {}'.format(min_losses))

        # ... and prepare the win/loss dictionary with their names.
        for name in strat_names:
            strats_dfs['win'][name] = None
            strats_dfs['loss'][name] = None

        # A little helper function to get a new (win, lose) DataFrame tuple for
        # a specific strategy
        def collect_strategy_data(key):
            df = next(strategy_generators[key])

            if max_margin is not None:
                df = df[df.open_margin <= max_margin]

            if df.shape[0] == 0:
                return None, None

            winning_indices = df.max_profit >= winning_profit

            # Randomize the leg order to make the model more robust
            if randomize_legs:
                randomize_legs_columns(df)

            return df[winning_indices], df[~winning_indices]

        for strat in strat_names:
            log('Collecting {}'.format(strat.replace('_', ' ')))

            while True:

                # Figure out which types of trades must be collected
                try:
                    current_wins = strats_dfs['win'][strat].shape[0]
                except AttributeError:
                    current_wins = 0
                try:
                    current_losses = strats_dfs['loss'][strat].shape[0]
                except AttributeError:
                    current_losses = 0

                enough_wins   = current_wins   >= win_pool_size
                enough_losses = current_losses >= loss_pool_size

                if enough_wins and enough_losses:
                    break

                strat_wins, strat_losses = collect_strategy_data(strat)
                if not enough_wins and strat_wins is not None:
                    strats_dfs['win'][strat] = pd.concat((
                        strats_dfs['win'][strat], strat_wins))
                if not enough_losses and strat_losses is not None:
                    strats_dfs['loss'][strat] = pd.concat((
                        strats_dfs['loss'][strat], strat_losses))

                try:
                    total_wins = strats_dfs['win'][strat].shape[0]
                except AttributeError:
                    total_wins = 0
                try:
                    total_losses = strats_dfs['loss'][strat].shape[0]
                except AttributeError:
                    total_losses = 0
                log('\twins: {:<8} ({:.1%})\tlosses: {:<8} ({:.1%})'.format(
                    total_wins, total_wins / win_pool_size,
                    total_losses, total_losses / loss_pool_size))

            # If desired, sort the winners such that they are in order of smallest
            # max_profit to largest max_profit.
            if hard_winners:
                strats_dfs['win'][strat].sort_values(
                    by='max_profit', ascending=True, inplace=True)
            strats_dfs['win'][strat] = strats_dfs['win'][strat][:min_wins]
            # If desired, sort the losers such that they are in order of largest
            # max_profit to smallest max_profit.
            if hard_losers:
                strats_dfs['loss'][strat].sort_values(
                    by='max_profit', ascending=False, inplace=True)
            strats_dfs['loss'][strat] = strats_dfs['loss'][strat][:min_losses]

    # Concat all of the DataFrames
    df = pd.concat(
        [d for d in strats_dfs['win'].values()] +
        [d for d in strats_dfs['loss'].values()]
    )

    # Save the final examples DataFrame if a directory was specified
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fpath = os.path.join(save_dir, str(uuid.uuid4()))
        df.to_pickle(fpath)

    return df

def calculate_fee(count=1, both_sides=True):
    fee = config.BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee / 100

def get_security_prices(ticker, start_dt, end_dt, frequency):
    qs = QuestradeSecurities()
    candles = qs.get_candlesticks(ticker, str(start_dt), str(end_dt), frequency)
    return pd.DataFrame(
        data=[c['open'] for c in candles],
        index=pd.to_datetime([c['end'] for c in candles]),
    )

def date_string_to_expiry(date_string):
    # All expiries are actually ay 4 pm on the day of expiry
    return datetime.strptime(date_string, '%Y-%m-%d') + timedelta(hours=16)

def get_eastern_tz(dt):
    transition = next(t for t in config.DST_TRANSITIONS if t.date() > dt.date())
    return timezone(timedelta(hours=-4 if transition.month > 9 else -5))

def add_eastern_tz(dt):
    return dt.replace(tzinfo=get_eastern_tz(dt))

def get_epoch_timestamp(dt):
    return (dt.astimezone(timezone.utc) - config.EPOCH).total_seconds()

def expiry_string_to_aware_datetime(date_string):
    return add_eastern_tz(date_string_to_expiry(date_string))
