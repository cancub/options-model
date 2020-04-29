#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
import datetime
import os
import sys
import threading
import time
import random

import config
from data_management import update_data_files, update_price_df
from questrade_options_api import QuestradeTickerOptions

TICKERS    = [x.upper() for x in list(set(sys.argv[1:]))]
NOW_DT     = datetime.datetime.now()

# The lock related to accessing or modifying the filesystem
FS_LOCK    = threading.Lock()

# The lock related to adding a price to the price dictionary for this datetime
PRICE_LOCK = threading.Lock()
prices = {'datetime': [NOW_DT]}

# The function to use in threading
def options_gofer(q_obj, ticker):
    def log(msg):
        if not config.QUIET:
            print('{:>5}: {}'.format(ticker, msg))

    log('starting')

    # Retrieve and store all the available metadata for the company
    # TODO: save this value to file to avoid one RTT
    q_obj.load_company(ticker)

    if config.MULTITHREADED:
        PRICE_LOCK.acquire()

    # Save the most recent price of the underlying security
    prices[ticker] = [q_obj.get_security_price()]

    if config.MULTITHREADED:
        PRICE_LOCK.release()

    log('retreiving options for each available expiry')

    options = q_obj.get_options()

    log('processing options into dataframes')

    # Process the expiries and their respective series
    for ex, series_data in options.items():
        log('processing {}'.format(ex))
        update_data_files(ticker, ex, series_data, NOW_DT, FS_LOCK)

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

# Update the price dataframe now that all the threads are done
update_price_df(prices)
