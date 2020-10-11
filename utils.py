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

    if expiry is not None:
        df = df.xs(expiry, level=1, drop_level=False)

    return df

def load_spreads(
    ticker,
    expiry,
    winning_profit=None,
    loss_win_ratio=None,
    refresh=False,
    verbose=False
):
    ticker_dir = os.path.join(config.SPREADS_DIR, ticker)
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

    out_dir = tp.collect_spreads(options_df,
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
    exp_dir = os.path.join(out_dir, expiry)
    subprocess.check_call(
        ['tar', '-C', exp_dir, '-cf', spreads_path] + os.listdir(exp_dir))

    shutil.rmtree(out_dir)

    return spreads_path

def apply_func_to_dfs_in_tarball(tarball_path, func):
    with tempfile.TemporaryDirectory(prefix='options_func') as tmpdir:
        file_list = extract_and_get_file_list(tarball_path, tmpdir)
        for f in file_list:
            fpath = os.path.join(tmpdir, f)
            # Let the function know the name of the file, just in case
            df = func(pd.read_pickle(fpath), f)
            df.to_pickle(fpath)
        subprocess.check_call(
            ['tar', '-C', tmpdir, '-cf', tarball_path] + file_list)

def apply_func_to_dfs_in_dir(df_dir, func):
    exp_dir = os.path.join(df_dir)
    for f in os.listdir(exp_dir):
        fpath = os.path.join(exp_dir, f)
        df = func(pd.read_pickle(fpath), fpath)
        df.to_pickle(fpath)

def extract_and_get_file_list(tarball_path, output_dir):
    files_bytes = subprocess.check_output(
        ['tar', '-C', output_dir, '-xvf', tarball_path])
    return sorted(files_bytes.decode('utf-8').strip().split('\n'))

def add_timestamp_columns(df):
    # Insert a colum which is the expiry in epoch timestamp form
    df.insert(0, 'expiry_timestamp', df.expiry.map(get_epoch_timestamp))

    # Insert a column which is the open_time in the form of an integer
    # representing the number of minutes until expiry
    df.insert(
        0,
        'seconds_to_expiry',
        df.apply(
            lambda x: x.expiry_timestamp - get_epoch_timestamp(x.open_time),
            axis = 1
        )
    )

def add_options_type_categories(df):
    option_val = 1 if df.iloc[0].leg1_type == 'C' else -1
    for i in range(1, config.TOTAL_LEGS + 1):
        df.insert(
            0,
            'leg{}_type_cat'.format(i),
            option_val if df['leg{}_strike'.format(i)].iloc[0] != 0 else 0
        )

def process_trades_df(df):
    add_timestamp_columns(df)
    add_options_type_categories(df)
    return sort_trades_df_columns(df)

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
            # Make sure that all DataFrames generated by this function are
            # uniform with respect to consituent columns and their order
            yield process_trades_df(pd.read_pickle(p))
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
    return int((dt.astimezone(timezone.utc) - config.EPOCH).total_seconds())

def expiry_string_to_aware_datetime(date_string):
    return add_eastern_tz(date_string_to_expiry(date_string))

def describe_wins(wins_df, winning_profit, ticker):
    expiry_options = {e: load_options(ticker, str(e.date()))
                        for e in wins_df.expiry.unique()}
    profit_details = {
        'open_to_first_profit': [],
        'first_profit_to_expiry': [],
        'profitable_at_expiry': [],
        'profitable_periods': [],
        'min_profitable_time': [],
        'median_profitable_time': [],
        'max_profitable_time': [],
        'total_profitable_time': []
    }

    one_min = timedelta(minutes=1)

    for index, row in wins_df.iterrows():
        print(index)

        row_open = row.open_time
        row_exp = row.expiry

        # Get the bid and ask prices for this specific expiry and option type.
        # Additionally, ignore any times at or before the open time and any
        # times after the expiry when considering closes.
        close_options = expiry_options[row_exp].xs(
            row.leg1_type, level=1
        ).loc[row_open + one_min: row_exp + one_min, ('bidPrice', 'askPrice')]

        credits = 0
        for i in range(1, config.TOTAL_LEGS + 1):
            strike = row['leg{}_strike'.format(i)]
            if strike == 0:
                continue
            price_df = close_options.xs(strike, level=1)
            leg_open_credit = row['leg{}_credit'.format(i)]

            if leg_open_credit > 0:
                # We sold this leg to start
                leg_close_credits = -price_df.askPrice.fillna(0.01)
            else:
                # We bought this leg to start
                leg_close_credits = price_df.bidPrice.fillna(0)

            credits += leg_close_credits + leg_open_credit

        # Sanity check to see that we arrived at the same conclusion as the
        # spreads collector
        try:
            assert(np.isclose(credits.max(), row.max_profit))
        except Exception:
            print(row)
            print(credits.max(), row.max_profit)
            raise

        profitable = ((credits > winning_profit)
                      | np.isclose(credits, winning_profit))

        # This function only accepts winning trades for the time being
        assert(profitable.sum() > 0)

        # Find first point of profitability and how it relates to the open and
        # expiry times
        first_profit = profitable.index[profitable][0]
        profit_details['open_to_first_profit'].append(first_profit - row_open)
        profit_details['first_profit_to_expiry'].append(row_exp - first_profit)

        # It's important to know that we could hold on to this trade and profit
        # just by letting it expire
        profitable_at_expiry = profitable[-1]
        profit_details['profitable_at_expiry'].append(profitable_at_expiry)

        # Find the transitions
        current_is_p = profitable[:-1].values
        next_is_p = profitable[1:].values
        transitions_up = profitable.index[1:][~current_is_p & next_is_p]
        transitions_down = profitable.index[1:][current_is_p & ~next_is_p]

        # Fill in the blanks depends on whether or not this was profitable at
        # the first possible sell point and/or if it was profitable at expiry
        if profitable[0]:
            transitions_up = sorted(
                transitions_up.append(pd.DatetimeIndex([row_open])))
        if profitable_at_expiry:
            transitions_down = sorted(
                transitions_down.append(pd.DatetimeIndex([row_exp])))

        # Find the periods in which this trade was profitable
        profitable_periods = []
        for i in range(len(transitions_up)):
            profitable_periods.append(transitions_down[i] - transitions_up[i])

        # Stats on profitable periods
        profit_details['profitable_periods'].append(len(profitable_periods))
        profit_details['min_profitable_time'].append(np.min(profitable_periods))
        profit_details['median_profitable_time'].append(
            np.median(profitable_periods))
        profit_details['max_profitable_time'].append(
            np.max(profitable_periods))
        profit_details['total_profitable_time'].append(
            np.sum(profitable_periods))

    return pd.concat((wins_df, pd.DataFrame(profit_details)), axis = 1)
