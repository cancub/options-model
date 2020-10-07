#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
import datetime
import itertools
import multiprocessing
import os
import pandas as pd

import config
from questrade_options_api import QuestradeTickerOptions

# Get the API object. No arguments implies we want to automatically reload and
# refresh the token.
QT = QuestradeTickerOptions()

# The seconds and microseconds are just clutter. Also set the timezone of the
# current time based on the server.
NOW_DT = datetime.datetime.now().replace(
    second=0, microsecond=0, tzinfo=QT.get_timezone())

# The function to use in threading
def options_gofer(q_obj, ticker):
    # Retrieve and store all the available metadata for the company
    # TODO: save this value to file to avoid one RTT
    q_obj.load_company(ticker)

    # Retrieve and the data related to the options of an underlying security
    ticker_options = q_obj.get_options()

    # Set up for building the DataFrame
    expiry_type_strikes = []
    index_names = ['datetime', 'expiry', 'type', 'strike']
    data = {n: [] for n in config.DATA_NAMES}

    for expiry, series in ticker_options.items():
        for sId, info in series.items():
            expiry_type_strikes.append((expiry, info['type'], info['strike']))
            for n in config.DATA_NAMES:
                data[n].append(info['data'][n])

    indices = ([a, b[0], b[1], b[2]] for a, b
                    in itertools.product((NOW_DT,), expiry_type_strikes))

    to_add_df = pd.DataFrame(
        data, index = pd.MultiIndex.from_tuples(indices, names=index_names))

    # Load up the existing DataFrame for this ticker and append this new data
    # to it
    ticker_path = os.path.join(config.STORAGE_DIR, ticker)
    try:
        ticker_df = pd.concat((pd.read_pickle(ticker_path), to_add_df))
    except FileNotFoundError:
        # This ticker doesn't exist yet, so this is all the data we have
        ticker_df = to_add_df

    # Save the new ticker DataFrame to file
    ticker_df.to_pickle(ticker_path)

if __name__ == '__main__':
    processes = []

    for ticker in config.TICKERS:
        p = multiprocessing.Process(
            target=options_gofer,
            args=(deepcopy(QT), ticker,)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
