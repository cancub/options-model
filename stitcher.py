import datetime as dt
from functools import wraps
from io import BytesIO
import numpy as np
import os
import pandas as pd
import re
import shutil
import subprocess

import trade_processing as tp
import utils

TRADES_DIR = 'trades'
STAGING_DIR = 'staging'
BACKUPS_DIR = 'backups'
PICKLES_DIR = 'testing/pickles'

def clean_staging_dir(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        os.mkdir(STAGING_DIR)
        try:
            return func(*args, **kwargs)
        finally:
            shutil.rmtree(STAGING_DIR)
    return wrapped_func

@clean_staging_dir
def load_data_from_tarball(tarball):
    subprocess.check_call(
        ['tar', '-C', STAGING_DIR, '-xf',
             os.path.join(os.path.abspath('.'), BACKUPS_DIR, tarball)]
    )
    prices = pd.read_pickle(
        os.path.join(STAGING_DIR, 'price')
    ).set_index('datetime')
    
    dataframes = {}

    for exp in (e for e in os.listdir(STAGING_DIR) if e != 'price'):
        dataframes[exp] = {}
        exp_dt = dt.datetime.strptime(exp, '%Y-%m-%d')
        exp_dir = os.path.join(STAGING_DIR, exp)
        for tik in (t for t in os.listdir(exp_dir) if '_' not in t):
            file_base = os.path.join(exp_dir, tik)

            # We need to load the times first to make sure we're not processing
            # useless data. So we collect the times first
            with open(file_base + '_times', 'r') as TF:
                dtimes = utils.get_basic_datetimes(
                    TF.read().split('\n')[:-1])

            # If the expiry datetime is _before_ the first polling time, it's
            # not much use to us. How do you make a bet on a known past event?
            if exp_dt < dtimes[0]:
                print(
                    'skipping {:>4} {} because it\'s before {}'.format(
                        tik, exp, dtimes[0])
                )
                continue

            data = np.load(file_base, allow_pickle=True)
            metadata = pd.read_pickle(
                file_base + '_meta').reset_index(drop=True)
            try:
                assert(data.shape[0] == 10)
                assert(data.shape[1] == len(dtimes))
                assert(data.shape[2] == len(metadata))
            except Exception as e:
                print('Error with {} {}:\n{}\nSkipping'.format(tik, exp, e))
                continue

            dataframes[exp][tik] = utils.process_options(data, metadata, dtimes)
    # Return price as well as dataframes
    return prices, dataframes

options = {}
prices = {}
for fname in os.listdir(BACKUPS_DIR):
    try:
        date, _ = fname.split('.tar.gz')
    except ValueError:
        continue
    print(fname)
    prices[date], options[date] = load_data_from_tarball(fname)

'''
The layers for the output data are
    1. expiry
    2. ticker
    3. Call/Put
    4. bid/ask
'''
final_options_dfs = {}
price_df = None
for tarball_date in sorted(options.keys()):
    # Piece together the prices
    new_price_df = prices[tarball_date]
    try:
        last_index = price_df.index[-1]
    except:
        # No price dataframe exists yet
        price_df = new_price_df
    else:
        to_add_df = new_price_df[last_index:]
        # Note that the equivalency of indices means that we may
        # need to skip the first entry
        if last_index in to_add_df.index:
            to_add_df = to_add_df.iloc[1:]

        price_df = pd.concat((price_df, to_add_df))

    # Walk through the expiries
    options_data = options[tarball_date]
    for exp, expiry_data in options_data.items():
        try:
            exp_dfs = final_options_dfs[exp]
        except KeyError:
            exp_dfs = final_options_dfs[exp] = {}
        for tik, ticker_data in expiry_data.items():
            last_time = None
            try:
                ticker_dfs = exp_dfs[tik]
            except KeyError:
                # No previous data means that this is the start
                exp_dfs[tik] = ticker_data
                continue

            # So now we have two dictionaries with ['C'/'P']['bid'/'ask']
            for otype in ('C', 'P'):
                for side in ('bid', 'ask'):
                    existing_df = ticker_dfs[otype][side]
                    new_df = ticker_data[otype][side]

                    # Find the date _after_ the last time index from the existing
                    # df and use that to add in the date from the new df.
                    last_index = existing_df.index[-1]
                    to_add_df = new_df[last_index:]

                    # Note that the equivalency of indices means that we may
                    # need to skip the first entry
                    if last_index in to_add_df.index:
                        to_add_df = to_add_df.iloc[1:]

                    ticker_dfs[otype][side] = pd.concat((existing_df, to_add_df))


PROCESSING_DIR = 'processing'
price_df.to_pickle(os.path.join(PROCESSING_DIR,'prices'))
'''
The layers for the output data are
    1. expiry
    2. ticker
    3. Call/Put
    4. bid/ask
'''
for exp, ticker_dfs in final_options_dfs.items():
    exp_dir = os.path.join(PROCESSING_DIR, 'single_legs', exp)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    for ticker, CP_dfs in ticker_dfs.items():
        for otype, bidask_dfs in CP_dfs.items():
            for bidask, df in bidask_dfs.items():
                df.to_pickle(
                    os.path.join(
                        exp_dir, '{}_{}{}'.format(ticker, otype, bidask))
                )
