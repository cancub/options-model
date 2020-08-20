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


def calculate_fee(count, both_sides=False):
    fee = BASE_FEE + count
    if both_sides:
        fee *= 2
    return fee
