import datetime as dt
from   functools import wraps
import itertools
import multiprocessing
import numpy as np
import os
import pandas as pd
import queue
import shutil
import subprocess

import config
import trade_processing as tp
import utils

STAGING_DIR = 'staging'
BACKUPS_DIR = 'backups'
PROCESSING_DIR = 'processing'

THREAD_COUNT = 8

def clean_staging_dir(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        os.mkdir(STAGING_DIR)
        try:
            return func(*args, **kwargs)
        finally:
            shutil.rmtree(STAGING_DIR)
    return wrapped_func

def worker(i, input_queue, output_queue):
    while True:
        try:
            exp, tik = input_queue.get(timeout=0.1)
        except queue.Empty:
            break

        file_base = os.path.join(STAGING_DIR, exp, tik)

        if not os.path.exists(file_base):
            continue

        exp_dt = dt.datetime.strptime(exp, '%Y-%m-%d')

        # We need to load the times first to make sure we're not processing
        # useless data. So we collect the times first
        with open(file_base + '_times', 'r') as TF:
            dtimes = utils.get_basic_datetimes(
                TF.read().split('\n')[:-1])

        # If the expiry datetime is _before_ the first polling time, it's
        # not much use to us. How do you make a bet on a known past event?
        if exp_dt < dtimes[0]:
            print('skipping {:>4} {} because it\'s before {}'.format(
                tik, exp, dtimes[0]))
            continue

        time_count = len(dtimes)
        data = np.load(file_base, allow_pickle=True)
        metadata = pd.read_pickle(
            file_base + '_meta').reset_index(drop=True)
        try:
            assert(data.shape[0] == 10)
            assert(data.shape[1] == time_count)
            assert(data.shape[2] == len(metadata))
        except Exception as e:
            print('Error with {} {}:\n{}\nSkipping'.format(tik, exp, e))
            continue

        data_reshape = data.reshape((len(config.DATA_NAMES), -1), order='F').T
        unflattened_index = itertools.product(
            dtimes,
            (exp,),
            metadata[['type','strike']].itertuples(index=False, name=None),
        )
        flattened = ([a, b, c[0], c[1]]
                        for a, b, c in unflattened_index)
        mindex = pd.MultiIndex.from_tuples(
            flattened, names=['datetime', 'expiry', 'type', 'strike'])
        new_df = pd.DataFrame(
            data_reshape,
            columns=config.DATA_NAMES,
            index=mindex,
        )
        output_queue.put((tik, new_df))

    print(i, 'done')

@clean_staging_dir
def load_data_from_tarball(tarball, skip_ranges=None):
    # TODO: just ignore certain time blocks if we already have them


    subprocess.check_call(
        ['tar', '-C', STAGING_DIR, '-xf',
             os.path.join(os.path.abspath('.'), BACKUPS_DIR, tarball)]
    )
    prices = pd.read_pickle(
        os.path.join(STAGING_DIR, 'price')
    ).set_index('datetime')
    prices.index = [x.replace(microsecond=0, second=0) for x in prices.index]

    exp_tik_q = multiprocessing.Queue()

    dataframes = multiprocessing.Queue()

    df_list = {}

    all_combos = list(itertools.product(
        (e for e in os.listdir(STAGING_DIR) if e != 'price'),
        config.TICKERS
    ))
    for c in all_combos:
        exp_tik_q.put(c)

    # worker(0, exp_tik_q, dataframes)

    processes = []

    for i in range(THREAD_COUNT):
        p = multiprocessing.Process(
            target=worker, args=(i, exp_tik_q, dataframes,))
        p.start()
        processes.append(p)

    # This is going to be a lot of buffered data in each of the individual
    # Queues, so it's better to avoid this by continually checking for data and
    # cleaning up processes at the same time
    while len(processes) > 0:
        try:
            tik, df = dataframes.get(block=False)
        except queue.Empty:
            pass
        else:
            try:
                df_list[tik].append(df)
            except KeyError:
                df_list[tik] = [df]
        processes_to_remove = []
        for p in processes:
            p.join(timeout=0)
            if not p.is_alive():
                processes_to_remove.append(p)
        processes = list(set(processes) - set(processes_to_remove))
    
    # Try a get from the Queue one last time, just in case they put() and exited
    # just after our last call to get()
    while dataframes.qsize() > 0:
        tik, df = dataframes.get(block=False)
        df_list[tik].append(df)

    # Return price as well as options data
    return prices, df_list

for fname in sorted(os.listdir(BACKUPS_DIR)):
    try:
        date, _ = fname.split('.tar.gz')
    except ValueError:
        continue
    print(fname)

    new_prices, new_options = load_data_from_tarball(fname)
    try:
        prices = pd.concat((prices,new_prices))
        for tik, df_list in new_options.items():
            options[tik] = pd.concat([options[tik]] + df_list)
    except:
        prices = new_prices
        options = {}
        for tik, df_list in new_options.items():
            options[tik] =  pd.concat(df_list)
    else:
        prices = prices.loc[~prices.index.duplicated(keep='first')]
        prices.to_pickle(os.path.join(PROCESSING_DIR, 'stock_prices'))
        for tik, df in options.items():
            options[tik] = options[tik].loc[
                ~options[tik].index.duplicated(keep='first')]
            options[tik].to_pickle(os.path.join(PROCESSING_DIR, tik))
