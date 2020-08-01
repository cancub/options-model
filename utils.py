import datetime as dt
import numpy as np
import re

BASE_FEE = 9.95

def get_basic_datetimes(times_strings):
    dt_match = re.compile(r'\d+-\d+-\d+ \d+:\d+')
    strptime_format = '%Y-%m-%d %H:%M'
    return np.array(list(map(
        lambda x: dt.datetime.strptime(
            re.match(dt_match, x).group(0), strptime_format),
        times_strings
    )))

def process_options(data, metadata, dtimes):
    result = {}
    for otype in ['C', 'P']:
        # Filter metadata for this option type
        type_meta = metadata[metadata['type'] == otype]

        # Get the data (bid, ask) for these strikes
        bid_df = pd.DataFrame(
            data[0, :, list(type_meta.index)].T,
            index=dtimes,
            columns=type_meta['strike']
        )
        ask_df = pd.DataFrame(
            data[1, :, list(type_meta.index)].T,
            index=dtimes,
            columns=type_meta['strike']
        )

        # Get the DataFrames for all combinations of bid/ask + open/all
        result[otype] = {'bid': bid_df, 'ask': ask_df}

    return result

def calculate_fee(count, both_sides=False):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee
