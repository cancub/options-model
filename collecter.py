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

# Globals
TICKERS                 = sys.argv[1:]
WINDOW_DAYS             = 90
STORAGE_DIR             = 'pickles'
META_COLUMNS            = ['symbolId', 'type', 'strike']
DATA_NAMES              = ['lastTradePrice', 'volume', 'volatility', 'delta', 'gamma',
                           'theta', 'vega', 'rho', 'openInterest']
NOW_DT                  = datetime.now()

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
    WINDOW_DAYS days (on fridays) years.
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
                (expiry + timedelta(days=WINDOW_DAYS)).isoformat(),
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

for ticker in TICKERS:
    print(ticker)
    # Make sure we have the proper storage directory for the ticker
    try:
        os.mkdir(os.path.join(STORAGE_DIR, ticker))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Get the api-related dict and load the API object
    with open(os.path.expanduser('~/.questrade.json'), 'r') as QF:
        qtrade_json = json.load(QF)
    q = Questrade(refresh_token=qtrade_json['refresh_token'])

    # Build up basic description of company
    company = None
    for c in q.symbols_search(prefix=ticker)['symbols']:
        if c['symbol'] == ticker:
            company = c
            break
    if company is None:
        continue
    code = company['symbolId']

    # Find the most recent info about the underlying security
    current_info = q.markets_quote(code)
    for info in current_info['quotes']:
        if info['symbol'] == ticker:
            current_info = info
            break
    update_price_df(ticker, current_info)

    # Starting from the upcoming friday, get the list of expiry dates
    #going out to the specified number of days
    expiries = []
    today = date.today()

    results = {}
    # Get all the data in one fell swoop for each expiry
    for ex in get_expiry_dates():
        results[str(ex)] = q.markets_options(
            filters=[{
                'underlyingId': code,
                'expiryDate': str(ex),
            }])['optionQuotes']

    # Process the expiries and their respective series
    for ex, series_data in results.items():
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
