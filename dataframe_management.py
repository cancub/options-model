import errno
import os
import pandas as pd

import config

def get_expiry_dataframes(ticker, expiry):
    # Get the dataframe from any available pickles.
    # Create them if they're not there
    expiry_dir = os.path.join(config.STORAGE_DIR, ticker, expiry)
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
        dataframes['meta'] = pd.DataFrame(columns=config.META_COLUMNS)

    for n in config.DATA_NAMES:
        try:
            dataframes[n] = pd.read_pickle(os.path.join(expiry_dir, n))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            dataframes[n] = pd.DataFrame({'datetime': []})

    return dataframes


def write_expiry_dataframes(ticker, expiry, dataframes):
    expiry_dir = os.path.join(config.STORAGE_DIR, ticker, expiry)
    for name, df in dataframes.items():
        df.to_pickle(os.path.join(expiry_dir, name))


def add_new_series(meta_df, series_data):
    # Add any series that appear in this new set of options data but not in
    # our set of previously-seen series
    seen_symbols = meta_df['symbolId'].values

    for opid, data in series_data.items():
        if opid not in seen_symbols:
            metadata_dict = {
                'symbolId': [opid],
                'type': [data['type']],
                'strike': [data['strike']],
            }
            meta_df = meta_df.append(pd.DataFrame(metadata_dict))
    return meta_df

def update_price_df(ticker, latest_price, now_datetime):
    # store the most recent price in the main price dataframe
    price_path = os.path.join(config.STORAGE_DIR, ticker, 'price')

    append_df = pd.DataFrame(
        {'datetime': [now_datetime], 'price': [latest_price]})
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
