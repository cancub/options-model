import errno
import numpy as np
import os
import pandas as pd

import config

def add_layers(array, count, axis=0, fill = 0):
    '''
    Add either certain number of either rows or columns with a specific fill
    value
    '''
    if axis not in (0,1):
        raise Exception('Layers can only be added as rows (0) or columns (1)')
    if not isinstance(np.nan, (int, float)):
        raise Exception('Fill must be int or float')

    depth, rows, cols = array.shape

    rows += (1-axis) * count
    cols += axis * count

    new_array = np.zeros((depth, rows, cols), dtype=array.dtype)

    if axis == 0:
        new_array[:, :-count, :] = array
        if fill != 0:
            new_array[:, -count:, :] = fill
    else:
        new_array[:, :, :-count] = array
        if fill != 0:
            new_array[:, :, -count:] = fill

    return new_array

def add_new_datapoint(array, new_datapoint):
    if array is None:
        # THe new datapoint is the _only_ datapoint so far
        new_array = new_datapoint
    else:
        # allocate memory for an array jsut big enough to fit the old array and
        # the new datapoint
        new_array = add_layers(array, 1, axis=0, fill=np.nan)
        new_array[:,-1:,:] = new_datapoint
    return new_array

def update_data_files(ticker, expiry, series, now_dt, fs_lock = None):
    '''
    series:
    {
        <symbolId>: {
            'type': ['C','P'],
            'strike': <float>,
            <data>,
        },
        ...
    },
    '''
    # Load up the existing objects from their respective files

    expiry_dir = os.path.join(config.STORAGE_DIR, expiry)

    if fs_lock is not None:
        fs_lock.acquire()
    try:
        os.mkdir(expiry_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    finally:
        if fs_lock is not None:
            fs_lock.release()

    base_path = os.path.join(expiry_dir, ticker)
    meta_path = base_path + '_meta'

    # The actual 3D data array
    data = None
    try:
        data = np.load(base_path, allow_pickle=True)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

    # Metadata dataframe
    try:
        meta_df = pd.read_pickle(meta_path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        meta_df = pd.DataFrame(columns=config.META_COLUMNS)

    # For storing the new data from series
    new_datapoint = np.zeros(
        (len(config.DATA_NAMES), 1, len(series)), dtype=np.float32)
    # Rather than trying to guess at how many previously unseen options we have
    # in series, make another array the same shape as the final product and add
    # all these specific options to this array and then copy them over
    unseen_array = np.copy(new_datapoint)

    # TODO: - treat the case of a new expiry-ticker combo differently (don't do 
    #         extra processing)
    #       - try for op in meta_df['symbolId'].toList() to see if its faster

    # TODO: add columns for hitherto unseen symbols

    # The list of options that we have seen before, with the order acting as the
    # index for 
    existing_ops = meta_df['symbolId'].tolist()

    # For storing the iformation about options we haven't seen before
    unseen_df_dict = {'symbolId': [], 'type': [], 'strike': []}
    unseen_count = 0

    for sId, info in series.items():
        array_to_add = np.array(
            [info['data'][col] for col in config.DATA_NAMES])
        try:
            new_datapoint[:, 0, existing_ops.index(sId)] = array_to_add
        except ValueError:
            # This is a symbol we haven't seen before, so prepare to add it to
            # both the data array _and_ the metadata DataFrame by collecting all
            # the relevant information for each.
            unseen_array[:, 0, unseen_count] = array_to_add
            unseen_count += 1
            unseen_df_dict['symbolId'].append(sId)
            for key in ('type', 'strike'):
                unseen_df_dict[key].append(info[key])

    # At this point we've added data for existing options and we have catalogued
    # the data options that we need to add. So let's take care of the latter.
    if unseen_count > 0:
        if data is None:
            # This is the creation event for this ticker-expiry, so _all_
            # options are previously-unseen options
            new_datapoint = unseen_array
        else:
            # Previous data already exists, so we need to make room for the new
            # series (setting all previous timeslots where it did not exist to
            # NaN)
            data = add_layers(data, unseen_count, axis=1, fill=np.nan)
            # Now insert the data for the unseen options at the end of the new
            # datapoint
            new_datapoint[:,:,-unseen_count:] = unseen_array[:,:,:unseen_count]

        # Now that we've added the data, we can update the meta dataframe
        meta_df.append(pd.DataFrame(unseen_df_dict)).to_pickle(meta_path)

    data = add_new_datapoint(data, new_datapoint)

    # Save the data
    with open(base_path, 'wb') as NF:
        np.save(NF, data)

    # Finally, write the latest time to the end of the csv
    with open(base_path + '_times', 'a') as TF:
        TF.write('{}\n'.format(now_dt))


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
            # We need to ignore index so that the index of each row corresponds
            # to the index of the respective option in the numpy dictionary
            meta_df = meta_df.append(
                pd.DataFrame(metadata_dict), ignore_index=True)
    return meta_df

def update_price_df(data_dict):
    # store the most recent price in the main price dataframe
    price_path = os.path.join(config.STORAGE_DIR, 'price')

    append_df = pd.DataFrame(data_dict)
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
