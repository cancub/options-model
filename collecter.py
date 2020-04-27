#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
import datetime
import errno
import json
import os
import pandas as pd
import sys
import threading
import time
import random

import config
from dataframe_management import (get_expiry_dataframes, write_expiry_dataframes,
                                  add_new_series, update_price_df)
from questrade_options_api import QuestradeTickerOptions

TICKERS          = sys.argv[1:]
NOW_DT           = datetime.datetime.now()

# The function to use in threading
def options_gofer(q_obj, ticker):
    def log(msg):
        if not config.QUIET:
            print('{:>5}: {}'.format(ticker, msg))

    log('starting')

    # Retrieve and store all the available metadata for the company
    q_obj.load_company(ticker)

    # Make sure we have the proper storage directory for the ticker
    try:
        os.mkdir(os.path.join(config.STORAGE_DIR, ticker))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Save the most recent price of the underlying security
    update_price_df(ticker, q_obj.get_security_price(), NOW_DT)

    log('retreiving options for each available expiry')

    options = q_obj.get_options()

    log('processing options into dataframes')

    # Process the expiries and their respective series
    for ex, series_data in options.items():
        # Collect all the dataframes
        dataframes = get_expiry_dataframes(ticker, ex)

        # We're just adding any previously-unseen series in the metadata
        # dataframe, i.e., we won't be looping, so pop this out and add
        # it back with the unseen series
        meta_df = dataframes.pop('meta')

        # Update each of the data dfs
        for metadata_name in dataframes.keys():
            log('processing {} for {}'.format(metadata_name, ex))

            data_dict = {k: v['data'][metadata_name] 
                         for k, v in series_data.items()}
            data_dict['datetime'] =  [NOW_DT]
            dataframes[metadata_name] = dataframes[metadata_name].append(
                pd.DataFrame(data_dict))

        dataframes['meta'] = add_new_series(meta_df, series_data)

        write_expiry_dataframes(ticker, ex, dataframes)

    log('complete')

# Get the API object. No arguments implies we want to automatically reload and
# refresh the token.
q = QuestradeTickerOptions()

threads = []

for ticker in TICKERS:
    if config.MULTITHREADED:
        t = threading.Thread(target=options_gofer, args=(deepcopy(q), ticker,))
        t.start()
        threads.append(t)
    else:
        options_gofer(q,ticker)

if config.MULTITHREADED:
    for t in threads:
        t.join()


