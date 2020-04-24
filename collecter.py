#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import errno
import json
import os
import pandas as pd
from questrade_api import Questrade
import sys
import threading
import time
import random

TICKERS          = sys.argv[1:]

# Some tickers have a bunch of expiry dates (e.g., SPY), others not so much.
# We want to have at least MINIMUM_EXPIRIES to work with, but we don't want to
# go overboard. So stop when we both:
#   - hit the minimum
#   - qualify for either maximum
MINIMUM_EXPIRIES = 10
MAXIMUM_DAYS     = 90
MAXIMUM_EXPIRIES = 30

STORAGE_DIR      = 'pickles'
META_COLUMNS     = ['symbolId', 'type', 'strike']
DATA_NAMES       = ['lastTradePrice', 'volume', 'volatility', 'delta', 'gamma',
                    'theta', 'vega', 'rho', 'openInterest']
NOW_DT           = datetime.now()



# Helper functions
def update_price_df(ticker, current_info):
    # store the most recent price in the main price dataframe
    last_price = current_info['lastTradePrice']
    price_path = os.path.join(STORAGE_DIR, ticker, 'price')

    append_df = pd.DataFrame({'datetime': [NOW_DT], 'price': [last_price]})
    try:
        price_df = pd.read_pickle(price_path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        price_df = append_df
    else:
        price_df = price_df.append(append_df, ignore_index=True)

    # We're done with this info now
    price_df.to_pickle(price_path)

def get_expiry_dates():
    '''
    Starting from the upcoming friday, get the list of expiry dates for the next
    two years. With this, we'll be able to get at least 10 valid expirydates for
    each ticker
    '''
    today = date.today()

    # Questrade requires a datetime with a specific format (iso + UTC-5)
    expiry = datetime(
        day=today.day,
        month=today.month,
        year=today.year,
        tzinfo=timezone(-timedelta(hours=5))
    )

    # Only fridays
    if expiry.isoweekday() != 5:
        expiry += timedelta(days=5-expiry.isoweekday())

    return pd.date_range(
                expiry.isoformat(),
                (expiry + timedelta(weeks=104)).isoformat(),
                freq='7D'
            )

def get_expiry_dataframes(ticker, expiry):
    # Get the dataframe from any available pickles.
    # Create them if they're not there
    expiry_dir = os.path.join(STORAGE_DIR, ticker, expiry)
    try:
        os.mkdir(expiry_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    dataframes = {}

    # Metadata
    try:
        dataframes['meta'] = pd.read_pickle(os.path.join(expiry_dir, 'meta'))
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        dataframes['meta'] = pd.DataFrame(columns = META_COLUMNS)

    for n in DATA_NAMES:
        try:
            dataframes[n] = pd.read_pickle(os.path.join(expiry_dir, n))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            dataframes[n] = pd.DataFrame({'datetime': []})

    return dataframes

def write_expiry_dataframes(ticker, expiry, dataframes):
    expiry_dir = os.path.join(STORAGE_DIR, ticker, expiry)
    for name, df in dataframes.items():
        df.to_pickle(os.path.join(expiry_dir, name))

def add_new_series(meta_df, series_data):
    # Add any series that appear in this new set of options data but not in
    # our set of previously-seen series
    seen_symbols = meta_df['symbolId'].values
    for sy in series_data:
        if sy['symbolId'] not in seen_symbols:
            # A new series (maybe the price drifted by quite a bit)
            # "symbol": "SPY24Apr20C195.00"
            info_str = sy['symbol']
            # We can't make any assumptions about the contents of
            # the string. So walk back until we find a C or P
            option_type_index = len(info_str) - 1
            while option_type_index > -1:
                if info_str[option_type_index].lower() in ['c', 'p']:
                    break
                option_type_index -= 1
            metadata_dict = {
                'symbolId': [sy['symbolId']],
                'type': [info_str[option_type_index].lower()],
                'strike': [float(info_str[option_type_index+1:])],
            }
            meta_df = meta_df.append(pd.DataFrame(metadata_dict))
    return meta_df



# Script

# Get the api-related dict and load the API object
with open(os.path.expanduser('~/.questrade.json'), 'r') as QF:
    qtrade_json = json.load(QF)
q = Questrade(refresh_token=qtrade_json['refresh_token'])

# Build the list fo expiry dates that we'll be searching for in each ticker
dates   = get_expiry_dates()
max_day = dates[0] + timedelta(days=MAXIMUM_DAYS)

# We're now going to make a series of per-ticker API calls. We speed this up by
# using a single thread for each ticker. It's possible that a thread will be
# wasted on a non-existent ticker and as such it *would* be better to thread
# by _action_ rather than ticker (e.g., get symbol, get price, get expiry). This
# could get kinda unweidly though, since we should also be threading along expiry
# lines.

# The function to use in threading
def options_gofer(q_if, ticker):
    print('{}: starting'.format(ticker))

    # we need a random seed for the time.sleep without collision
    random.seed(sum((ord(ch) for ch in ticker)))

    # Intial temporal staggering of threads to help avoid simultaneous requests
    time.sleep(random.random())

    # Build up basic description of company
    company = None
    for c in q_if.symbols_search(prefix=ticker)['symbols']:
        if c['symbol'] == ticker:
            company = c
            break

    # Don't create diddly if we weren't able to find the company
    if company is None:
        raise Exception(
            'Could not get any information for {}. Exiting.'.format(ticker))


    # We know we've got the right company, so store the questrade code in order
    # to request it later
    code = company['symbolId']

    # Make sure we have the proper storage directory for the ticker
    try:
        os.mkdir(os.path.join(STORAGE_DIR, ticker))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Find the most recent info about the underlying security
    current_info = None
    response = q_if.markets_quote(code)
    for info in response['quotes']:
        if info['symbol'] == ticker:
            current_info = info
            break

    if current_info is None:
        raise Exception(
            'Could not find current quote for {}. Exiting.'.format(ticker))

    update_price_df(ticker, current_info)

    # Get all the data in one fell swoop for each valid expiry in our range
    expiries        = {}
    expiry_count    = 0

    for ex in dates:
        while True:
            # Another staggering
            time.sleep(random.random())

            r = q.markets_options(
                filters=[{
                    'underlyingId': code,
                    'expiryDate': ex.isoformat(),
                }])
            if 'optionQuotes' in r.keys():
                break
            elif 'code' in r.keys() and r['code'] == 1006:
                continue
            else:
                raise Exception('Dunno what this is: {}'.format(r))


        series = r['optionQuotes']
        if len(series) == 0:
            # Nothing to see here, move along
            continue

        expiries[ex.isoformat()] = list(series)
        expiry_count += 1

        print('{}: got valid expiry {}'.format(ticker, ex))

        # Check to see if we've got at least the number of expiries we wanted
        # AND we're etiher past the specified maximum number of days or now have
        # the maximum number of expiries.
        if expiry_count >= MINIMUM_EXPIRIES:
            if ( (ex >= max_day) or (expiry_count == MAXIMUM_EXPIRIES) ):
                break

    print('{}: processing series for each expiry'.format(ticker))
    # Process the expiries and their respective series
    for ex, series_data in expiries.items():
        # Collect all the dataframes
        dataframes = get_expiry_dataframes(ticker, ex)

        # We're just adding any previously-unseen series in the metadata
        # dataframe, i.e., we won't be looping, so pop this out and add
        # it back with the unseen series
        meta_df = dataframes.pop('meta')

        # Update each of the data dfs
        for metadata_name in dataframes.keys():
            data_dict = {s['symbolId']: s[metadata_name] for s in series_data}
            data_dict['datetime'] =  [NOW_DT]
            dataframes[metadata_name] = dataframes[metadata_name].append(
                pd.DataFrame(data_dict))

        dataframes['meta'] = add_new_series(meta_df, series_data)

        write_expiry_dataframes(ticker, ex, dataframes)

    print('{}: complete'.format(ticker))

threads = []

for ticker in TICKERS:
    t = threading.Thread(target=options_gofer, args=(deepcopy(q), ticker,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
