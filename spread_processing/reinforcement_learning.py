from multiprocessing import Process, Value, Queue
import numpy as np
import pandas as pd
import queue

import config
import utils

# ============================== Vertical spreads ==============================

def vert_generator(df, strikes, max_margin=None):

    def affordable_opens(series):
        try:
            return (sum(series) < max_margin).any()
        except TypeError:
            return False

    leg1_strikes = strikes[:]
    np.random.shuffle(leg1_strikes)

    while True:

        try:
            leg1_strike = leg1_strikes.pop(0)
        except IndexError:
            break
        else:
            if len(leg1_strikes) == 0:
                # We can't compare a leg to itself
                break

        leg1_df = df.xs(leg1_strike, level=1)

        opens = [leg1_df.askPrice]
        if not affordable_opens(opens):
            continue

        # Use the remaining strikes as the source for the second leg
        leg2_strikes = leg1_strikes[:]

        while True:
            try:
                leg2_strike = leg2_strikes.pop(0)
            except IndexError:
                break

            leg2_df = df.xs(leg2_strike, level=1)

            if not affordable_opens(opens + [leg2_df.bidPrice]):
                continue

            # Ok, we have a long leg and a short leg that have at least one
            # open that is affordable. So we can continue on to build a
            # DataFrame representing all of the time points for this strategy.
            # Start by making a copy of each and then renaming the columns to
            # reflect their role in the strategy.
            orig_columns = leg1_df.columns
            long_df = leg1_df.rename(
                columns={c: 'long1_' + c for c in orig_columns})
            short_df = leg2_df.rename(
                columns={c: 'short1_' + c for c in orig_columns})


            # Combine these legs to build the strategy.
            try:
                strat_df = pd.concat((long_df, short_df), axis=1)
            except ValueError:
                # This stems from there being a duplicate in the indices.
                continue

            # There are likely to be timepoints for which we only have data
            # for one of the legs, so we make sure to fill the strike columns
            # with the respective values and then set everything else to 0.
            strat_df.insert(0, 'long1_strike', leg1_strike)
            strat_df.insert(0, 'short1_strike', leg2_strike)

            yield leg1_strike, leg2_strike, strat_df


def bfly_generator(
    base_df,
    vert_strat_df,
    mid_strike,
    strikes,
    long,
    max_margin=None
):

    def affordable_opens(series):
        try:
            return (sum(series) < max_margin).any()
        except TypeError:
            return False

    if long:
        # This is a long butterfly, so the duplicate strike is the short
        # strike and the highest strike is a long strike.
        mid_prefix = 'short1'
        mid_margin_col = mid_prefix + '_bidPrice'
        high_prefix = 'long2_'
        high_margin_col = 'askPrice'
    else:
        # This is a short butterfly, so the duplicate strike is the long
        # strike and the highest strike is a short strike.
        mid_prefix = 'long1'
        mid_margin_col = mid_prefix + '_askPrice'
        high_prefix = 'short2_'
        high_margin_col = 'bidPrice'

    opens = [vert_strat_df.long1_askPrice
                + vert_strat_df.short1_bidPrice
                + vert_strat_df[mid_margin_col]]

    if not affordable_opens(opens):
        return

    # Duplicate middle strike and give it column names to identify it as an
    # independent leg.
    rename_dict = {c: c.replace('1', '2')
                    for c in vert_strat_df.columns
                    if mid_prefix in c}
    three_df = pd.concat(
        (
            vert_strat_df,
            vert_strat_df[rename_dict.keys()].rename(columns=rename_dict)
        ),
        axis=1
    )

    # Build a dictionary to use for filling in empty strike values after all
    # the concats are done.
    mid_strike_col = mid_prefix + '_strike'
    fill_dict = {
        'long1_strike': three_df.long1_strike.min(),
        'short1_strike': three_df.long1_strike.min(),
        mid_strike_col: three_df[mid_strike_col].min()
    }

    high_strikes = [s for s in strikes if s > mid_strike]

    while True:
        try:
            high_strike = high_strikes.pop(0)
        except IndexError:
            break

        high_df = base_df.xs(high_strike, level=1)

        if not affordable_opens(opens + [high_df[high_margin_col]]):
            # Too rich for our blood.
            continue

        # Ok, we have a butterfly with at least one open that is affordable.
        # Use what we have so far to build a DataFrame representing all of the
        # time points for this strategy.
        try:
            strat_df = pd.concat(
                (
                    three_df,
                    high_df.rename(
                        columns={c: high_prefix + c for c in high_df.columns}
                    )
                ),
                axis=1
            )
        except ValueError:
            # This stems from there being a duplicate in the indices.
            continue

        strat_df.insert(0, high_prefix + 'strike', high_strike)

        # There are likely to be timepoints for which we only have data for one
        # of the legs, so we make sure to fill the strike columns with the
        # respective values.
        strat_df.fillna(fill_dict)

        yield strat_df

def option_type_generator(
    df,
    prices_series,
    vertical=True,
    butterfly=True,
    window_percent=0.015,
    max_margin=config.MARGIN
):

    if butterfly:
        # These will be used to restrict ourselves to a certain subset of
        # trades that are centered somewhere around the current strike price.
        window = prices_series * window_percent

    strikes = sorted(df.index.unique(level=1).to_list())

    for long_s, short_s, vert_df in vert_generator(df, strikes, max_margin):

        if vertical:
            # Continue building the completed DataFrame with the strikes for
            # (non-existent) legs 3 and 4.
            out_df = vert_df.astype(np.float32)
            for c in out_df.columns:
                out_df[c.replace('1', '2')] = 0

            yield out_df

        if butterfly:
            # Use the vertical dataframe as the starter to build all acceptable
            # butterflies.

            # Determine the middle strike which will be duplicated.
            mid_strike = max(long_s, short_s)

            # Was this strike ever in the specified window?
            low_enough = (-window + mid_strike) <= prices_series
            high_enough = (window + mid_strike) >= prices_series
            if not (low_enough & high_enough).any():
                # No, it was not.
                continue

            # Let the generator know which kind of butterfly it's dealing with:
            # long or short.
            long = short_s == mid_strike

            for bfly_df in bfly_generator(
                    df, vert_df, mid_strike, strikes, long, max_margin):
                yield bfly_df

def expiry_spreads_generator(
    expiry,
    expiry_df,
    prices,
    vertical=True,
    butterfly=True,
    window_percent=0.015,
    max_margin=config.MARGIN
):
    # Build a metadata DataFrame that will be concat'd with each of the
    # DataFrames output by the generators.
    meta_df = pd.DataFrame(prices)

    # Get the expiry as an aware datetime at the 4 pm closing bell.
    expiry_dt = utils.expiry_string_to_aware_datetime(expiry)

    eday = expiry_dt.day
    ewday = expiry_dt.weekday()
    eisowday = expiry_dt.isoweekday()

    # Insert a column which is the open_time in the form of an integer
    # representing the number of minutes until expiry
    def get_minutes_to_expiry(x):
        exp_ts = utils.get_epoch_timestamp(expiry_dt)
        cur_ts = utils.get_epoch_timestamp(x.name)
        return (exp_ts - cur_ts) / 60

    meta_df.insert(
        0,
        'minsToExpiry',
        meta_df.apply(get_minutes_to_expiry, axis=1)
    )

    # min-max expiry day of week (min = 1, max = 5)
    meta_df.insert(0, 'expiryDoW', (eisowday - 1) / 4)

    # min-max expiry week of month (min = 1, max = 5).
    meta_df.insert(
        0,
        'expiryWoM',
        (int(np.floor((eday - ewday + 3.9) / 7) + 1) - 1) / 4
    )

    otype_series = {}
    option_type_gens = {}

    for option_type in ('C', 'P'):

        option_type_gens[option_type] = option_type_generator(
            expiry_df.xs(option_type, level=1),
            prices,
            vertical,
            butterfly,
            window_percent,
            max_margin
        )

        otype_series[option_type] = pd.Series(
            index=meta_df.index,
            data=1 if option_type == 'C' else 0,
            name='optionType'
        )

    while len(option_type_gens) > 0:
        # Walk through each of the remaining expiries to get a new strategy
        # from each.
        for option_type, gen in option_type_gens.items():
            try:
                # Collect a new strategy for this option type.
                df = next(gen)

                # Add in the last few columns.
                indices = df.index
                final_df = pd.concat(
                    (
                        meta_df.loc[indices,:],
                        otype_series[option_type][indices],
                        df
                    ),
                    axis=1
                )

                # Make sure we only consider time before the expiry.
                yield final_df[final_df.index <= expiry_dt]

            except StopIteration:
                # There are no more trades for this expiry.
                del option_type_gens[expiry]

def spreads_generator(
    options_df,
    vertical=True,
    butterfly=True,
    window_percent=0.008,
    gen_processes=4,
    queue_size=32,
    max_margin=config.MARGIN
):

    def worker(expiry_generators, q):
        while len(expiry_generators) > 0:
            # Walk through each of the remaining expiries to get a new strategy
            # from each.
            for expiry, gen in expiry_generators.items():
                try:
                    # Provide a new strategy for this expiry.
                    df = next(gen)
                    q.put(df)
                except StopIteration:
                    # There are no more trades for this expiry.
                    del expiry_generators[expiry]

    # This is just filler right now.
    options_df.drop('symbolId', axis=1, inplace=True)

    # Get the prices of the stock at each time in this input dataframe.
    prices = options_df.pop('stock_price').droplevel(level=[1, 2, 3])
    prices = prices[~prices.index.duplicated(keep='first')].sort_index()
    # Work with the naming standard that has been set.
    prices.name = 'stockPrice'

    exp_count = 0
    exp_gens = {}
    for expiry in sorted(options_df.index.unique(level=1)):
        exp_count += 1
        exp_gens[expiry] = expiry_spreads_generator(
            expiry,
            options_df.xs(expiry, level=1),
            prices,
            vertical,
            butterfly,
            window_percent,
            max_margin
        )

    # Divide the expiries up evenly among the processes.
    proc_expiries = []
    base_expiries_per_proc = len(exp_gens) // gen_processes

    # There aren't enough expiries for all processes to get at least one.
    if base_expiries_per_proc == 0:
        gen_processes = len(exp_gens)
        base_expiries_per_proc = 1

    # The first pass through.
    for _ in range(gen_processes):
        to_add = {}
        for _ in range(base_expiries_per_proc):
            exp_count -= 1
            exp, gen = exp_gens.popitem()
            to_add[exp] = gen

        proc_expiries.append(to_add)
    
    # The second pass through for the remainder.
    for i in range(exp_count):
        exp, gen = exp_gens.popitem()
        proc_expiries[i][exp] = gen

    output_queues = {}
    for i in range(gen_processes):
        
        q = Queue(maxsize=queue_size)
        p = Process(target=worker, args=(proc_expiries[i], q))
        output_queues[p] = q
        p.start()

    while len(output_queues) > 0:
        for p, q in output_queues.items():
            try:
                yield q.get(timeout=0.1)
            except queue.Empty:
                if not p.is_alive():
                    # This thread is done, so stop checking its queue.
                    output_queues.pop(p)
                    p.join()


