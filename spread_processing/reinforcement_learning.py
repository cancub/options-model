import numpy as np
import pandas as pd

import config
import utils

# ============================== Vertical spreads ==============================

def vertical_spread_generator(
    all_strikes,
    strike_dfs,
    max_margin=None
):

    def affordable_opens(series):
        try:
            return (sum(series) < max_margin).any()
        except TypeError:
            return False

    for leg1_strike in all_strikes:

        leg1_df = strike_dfs[leg1_strike]

        opens = [leg1_df.askPrice]
        if not affordable_opens(opens):
            continue

        leg2_strikes = all_strikes[all_strikes != leg1_strike]

        for leg2_strike in leg2_strikes:

            leg2_df = strike_dfs[leg2_strike]

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
            strat_df.long1_strike = leg1_strike
            strat_df.short1_strike = leg2_strike

            yield strat_df

def butterfly_spread_generator(
    vert_strat_df,
    long,
    high_strike_dfs,
    max_margin=None,
):

    def affordable_opens(series):
        try:
            return (sum(series) < max_margin).any()
        except TypeError:
            return False

    if long:
        # This is a long butterfly, so the duplicate strike is the short
        # strike and the highest strike is a long strike.
        middle_prefix = 'short1'
        middle_margin_col = middle_prefix + '_bidPrice'
        highest_prefix = 'long2_'
        highest_margin_col = 'askPrice'
    else:
        # This is a short butterfly, so the duplicate strike is the long
        # strike and the highest strike is a short strike.
        middle_prefix = 'long1'
        middle_margin_col = middle_prefix + '_askPrice'
        highest_prefix = 'short2_'
        highest_margin_col = 'bidPrice'

    opens = [vert_strat_df.long1_askPrice
                + vert_strat_df.short1_bidPrice
                + vert_strat_df[middle_margin_col]]

    if not affordable_opens(opens):
        return

    # Duplicate middle strike and give it column names to identify it as an
    # independent leg.
    rename_dict = {c: c.replace('1', '2')
                for c in vert_strat_df.columns
                if middle_prefix in c}
    mid_df = vert_strat_df[rename_dict.keys()].rename(columns=rename_dict)

    for high_df in high_strike_dfs:
        if not affordable_opens(opens + [high_df[highest_margin_col]]):
            # Too rich for our blood.
            continue

        # Ok, we have a butterfly with at least one open that is affordable.
        # Use what we have so far to build a DataFrame representing all of the
        # time points for this strategy.
        try:
            strat_df = pd.concat(
                (
                    vert_strat_df,
                    mid_df,
                    high_df.rename(
                        columns={c: highest_prefix + c for c in high_df.columns}
                    )
                ),
                axis=1
            )
        except ValueError:
            # This stems from there being a duplicate in the indices.
            continue

        # There are likely to be timepoints for which we only have data for one
        # of the legs, so we make sure to fill the strike columns with the
        # respective values.
        for side in ('long', 'short'):
            for leg in ('1', '2'):
                col = side + leg + '_strike'
                vals = strat_df[col]
                if vals.isna().any():
                    # Min retrieves the non-nan value of the strike.
                    strat_df[col] = vals.min()

        # Set everything else to 0 and now we have everything we need to
        # output.
        yield strat_df

def option_type_spreads_generator(
    option_type,
    option_type_df,
    prices_series,
    vertical=True,
    butterfly=True,
    window_percent=0.015,
    max_margin=config.MARGIN
):

    if butterfly:
        # These will be used to restrict ourselves to a certain subset of
        # trades that are centered somewhere around the current strike
        # price.
        window = prices_series * window_percent

    # Convert the strike index to a column.
    option_type_df.reset_index('strike', inplace=True)

    # Build the dictionary of individual strikes data.
    all_strikes = option_type_df.strike.unique()
    np.random.shuffle(all_strikes)
    strike_dfs = {}
    for s in all_strikes:
        strike_dfs[s] = (
            option_type_df[option_type_df.strike == s].drop(
                'symbolId', axis=1)
        )

    vert_gen = vertical_spread_generator(
        all_strikes,
        strike_dfs,
        max_margin
    )

    for vert_df in vert_gen:

        if vertical:
            # Continue building the completed DataFrame with the
            # strikes for (non-existent) legs 3 and 4.
            out_df = vert_df.astype(np.float32)
            for c in out_df.columns:
                out_df[c.replace('1', '2')] = 0

            yield out_df

        if butterfly:
            # Use the vertical dataframe as the starter to build all
            # acceptable butterflies.

            # Determine the middle strike which will be duplicated.
            first = vert_df.iloc[0]
            middle_strike = max(
                first.long1_strike, first.short1_strike)

            # Was this strike ever in the specified window?
            low_enough = (-window + middle_strike) <= prices_series
            high_enough = (window + middle_strike) >= prices_series
            if not (low_enough & high_enough).any():
                # No, it was not.
                continue

            # There was at least one point in the history of this
            # strategy that it existed in the window we've specified.
            # Get all of the strikes which may be used for the highest
            # strike to complete the strategy.
            high_strikes = all_strikes[all_strikes > middle_strike]

            # Let the generator know which kind of butterfly it's
            # dealing with: long or short.
            long = vert_df.iloc[0].short1_strike == middle_strike

            butt_gen = butterfly_spread_generator(
                vert_df,
                long,
                [strike_dfs[s] for s in high_strikes],
                max_margin,
            )

            for butt_df in butt_gen:
                yield butt_df.astype(np.float32)

def expiry_spreads_generator(
    expiry,
    expiry_df,
    vertical=True,
    butterfly=True,
    window_percent=0.015,
    max_margin=config.MARGIN
):

    # Get the expiry as an aware datetime at the 4 pm closing bell.
    expiry_dt = utils.expiry_string_to_aware_datetime(expiry)

    # Get the prices of the stock at each time in this input dataframe.
    prices_series = expiry_df.groupby(level=[0])['stock_price'].first()

    # This is really a value for just the final DataFrame. No need to
    # include it here.
    expiry_df.drop('stock_price', axis=1, inplace=True)
    # Work with the naming standard that has been set.
    prices_series.name = 'stockPrice'
    prices_series = prices_series.astype(np.float32)

    option_type_gens = {}

    for option_type in ('C', 'P'):

        option_type_gens[option_type] = option_type_spreads_generator(
            option_type,
            expiry_df.xs(option_type, level=1),
            prices_series,
            vertical,
            butterfly,
            window_percent,
            max_margin
        )

    while len(option_type_gens) > 0:
        # Walk through each of the remaining expiries to get a new strategy
        # from each.
        for option_type, gen in option_type_gens.items():
            try:
                # Collect a new strategy for this option type.
                df = next(gen)

                # Add in the last few columns.
                df.insert(0, 'expiry', expiry_dt)
                df.insert(0, 'optionType', option_type)
                final_df = pd.concat((prices_series[df.index], df), axis=1)
                yield final_df

            except StopIteration:
                # There are no more trades for this expiry.
                del option_type_gens[expiry]

def spreads_generator(
    options_df,
    vertical=True,
    butterfly=True,
    window_percent=0.015,
    max_margin=config.MARGIN
):

    exp_gens = {}

    # Set things up so that we never see a trade from the same expiry twice as
    # long as there are still trades for each expiry.
    for expiry in sorted(options_df.index.unique(level=1)):
        exp_gens[expiry] = expiry_spreads_generator(
            expiry, 
            options_df.xs(expiry, level=1).copy(),
            vertical,
            butterfly,
            window_percent,
            max_margin
        )

    while len(exp_gens) > 0:
        # Walk through each of the remaining expiries to get a new strategy
        # from each.
        for expiry, gen in exp_gens.items():
            try:
                # Provide a new strategy for this expiry.
                yield next(gen)
            except StopIteration:
                # There are no more trades for this expiry.
                del exp_gens[expiry]


