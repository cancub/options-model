import logging
from   multiprocessing import Process, Value, Queue
import numpy as np
import pandas as pd
import queue

import config
import utils

HEADER_TEMPLATE = '{name:<15} {id:>2}:'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

# ============================== Vertical spreads ==============================

def vertical_spread_worker(
    id,
    strikes_q,
    butterfly_q,
    output_q,
    my_counter,
    option_type_df,
    bid_df,
    ask_df,
    prices_df,
    get_max_profit=True,
    max_margin=None,
    verbose=False
):
    def log(message):
        if not verbose: return
        logging.debug(
            '{hdr} {msg}'.format(
                hdr=HEADER_TEMPLATE.format(name='CP_COLLECTOR', id=id),
                msg = message
            ))

    log('START')

    # Questrade needs to take its cut too
    trade_fees = (config.BASE_FEE + 2) * 2 / 100

    all_strikes = ask_df.columns

    while True:
        # Grab a strike for the leg we will be longing
        try:
            leg1_strike = strikes_q.get(timeout=1)
        except queue.Empty:
            break

        close_credits = None

        leg2_strikes = all_strikes[all_strikes != leg1_strike]

        leg1_ask_s = ask_df[leg1_strike]
        leg1_bid_s = bid_df[leg1_strike]

        # The second leg cannot include the first leg strike
        leg2_bid_df = bid_df[leg2_strikes]
        leg2_ask_df = ask_df[leg2_strikes]

        # Figure out how much margin we'd need for this trade.
        open_margins = leg2_bid_df.add(leg1_ask_s, axis='rows')

        # Pretend that the too-expensive trades don't exist
        if max_margin is not None:
            open_margins[open_margins > max_margin] = np.nan

        base_df = open_margins.stack(
            level=0,
            dropna=False
        ).to_frame(name='open_margin')

        if get_max_profit:
            # Do some magic to get the maximum profits. Open credits is simple:
            # subtract the amount we pay for the first leg from the amount we
            # receive from the second.
            # NOTE: leave NaN in place to symbolize that this is not a viable
            #       trade. These trades will be removed at the end.
            open_credits = leg2_bid_df.sub(leg1_ask_s, axis='rows')

            # When looking at the close credits, make sure to replace NaN with 0
            # on the close sides, because NaN implies the trade was worthless.
            # This is good for leg 2 (since we buy it back for nothing at this
            # timepoint) and bad for leg 1 (since we can't sell it at this time
            # point).
            # NOTE: make this 0.01 for the leg 2 side since if we want to close
            #       early, we'd actually need to sell it for something.
            close_credits = (-leg2_ask_df.fillna(0.01)).add(
                                leg1_bid_s.fillna(0), axis='rows')

            # Get the maximum profit less fees
            max_profits = open_credits \
                          - trade_fees \
                          + utils.forward_looking_maximum(close_credits)

            # Insert the maximum profit Series to the output DataFrame.
            base_df.insert(
                1,
                'max_profit',
                max_profits.stack(level=0, dropna=False)
            )

        # Since we left the NaNs in place for the open credits, we can now strip
        # out the trades that we can't actually open because of:
        #   credits -> one or both sides weren't there to open, or
        #   margin  -> the combination was too pricey
        base_df.dropna(inplace=True)

        # If there's nothing left, then we can just skip this leg 1 strike
        total_trades = base_df.shape[0]
        if total_trades == 0:
            log('count({}) = 0'.format(int(leg1_strike)))
            continue

        # Bring all of the indices into the dataframe to be columns
        base_df.reset_index(level=[0, 1], inplace=True)

        # Update the column names to be more descriptive.
        rename_dict = {
            'datetime': 'open_time',
            'strike': 'leg2_strike',
        }
        base_df = base_df.rename(columns=rename_dict)

        # Add in the leg 1 strike
        leg1_strikes = np.full(total_trades, leg1_strike)
        base_df.insert(0, 'leg1_strike', leg1_strikes)

        all_open_times = base_df.open_time
        leg2_strikes = base_df.leg2_strike

        leg1_meta = option_type_df.loc[
            zip(all_open_times, leg1_strikes)].reset_index(drop=True)
        leg2_meta = option_type_df.loc[
            zip(all_open_times, leg2_strikes)].reset_index(drop=True)

        # Drop the bidPrice in the first leg and ask price in the second.
        # Additionally, we only want to store the stock price column once, so
        # pop it from the first leg for storage and drop it in the second leg.
        prices_series = leg1_meta.pop('stock_price')
        leg1_meta.drop(['bidPrice'], axis=1, inplace=True)
        leg2_meta.drop(['askPrice', 'stock_price'], axis=1, inplace=True)

        # Any models will need to know if we are buying or selling this leg, so
        # add in this column (BUY = 1, SELL = -1).
        leg1_meta.insert(0, 'action', 1)
        leg2_meta.insert(0, 'action', -1)

        # Change the other name to simply "credit." After inverting the value
        # for the ask price above, it's obvious whether it's a bid or an ask
        # price
        leg1_meta.rename(columns={'askPrice': 'credit'}, inplace=True)
        leg2_meta.rename(columns={'bidPrice': 'credit'}, inplace=True)

        orig_names = leg1_meta.keys()

        # We need to rename each of the columns
        leg1_meta.rename(
            columns={k: 'leg1_' + k for k in orig_names}, inplace=True)
        leg2_meta.rename(
            columns={k: 'leg2_' + k for k in orig_names}, inplace=True)

        # A butterfly spread can be built from these first two legs. Construct
        # the base dataframe (that is, would the empty legs 3 and 4) to give to
        # the butterfly spread worker.
        base_df = pd.concat(
            (
                base_df,
                prices_series,
                leg1_meta,
                leg2_meta,
            ),
            axis=1
        )

        # Provide the butterfly worker with everything it needs to do further
        # processing.
        butterfly_q.put((
            base_df.copy(),
            leg1_bid_s.copy(),
            leg1_ask_s.copy(),
            leg2_bid_df.copy(),
            leg2_ask_df.copy(),
            open_margins.copy(),
            open_credits.copy(),
            close_credits.copy(),
        ))

        # Continue building the completed DataFrame wiht the strikes for
        # (non-existent) legs 3 and 4.
        base_df.insert(0, 'leg3_strike', np.zeros(total_trades))
        base_df.insert(0, 'leg4_strike', np.zeros(total_trades))

        # The model will be expecting 4 legs, so fill in the remainder with 0s
        leg3_meta = pd.DataFrame(
            data=np.zeros(leg1_meta.shape),
            columns=['leg3_' + k for k in orig_names]
        )
        leg4_meta = pd.DataFrame(
            data=np.zeros(leg1_meta.shape),
            columns=['leg4_' + k for k in orig_names]
        )

        log('count({}) = {}'.format(int(leg1_strike), total_trades))

        # Now we have everything we need to output.
        output_q.put(
            pd.concat(
                (
                    base_df,
                    leg3_meta,
                    leg4_meta
                ),
                axis=1
            )
        )

    log('COMPLETE')

    # Signal that we are done.
    with my_counter.get_lock():
        my_counter.value -= 1

def butterfly_spread_worker(
    id,
    input_q,
    output_q,
    prev_counter,
    my_counter,
    option_type_df,
    bid_df,
    ask_df,
    prices_df,
    max_margin=None,
    verbose=False
):
    def log(message, long=None):
        if not verbose: return
        side = ''
        if long is not None:
            side = 'LONG' if long else 'SHORT'

        logging.debug(
            '{hdr} {msg} {side}'.format(
                hdr=HEADER_TEMPLATE.format(name='BFLY_COLLECTOR', id=id),
                msg = message,
                side=side
            ))

    log('START')

    all_strikes = ask_df.columns

    # Questrade needs to take its cut too.
    trade_fees = (config.BASE_FEE + 4) * 2 / 100

    # The percent of the current price on either side of the current price that
    # we will consider when collecting butterflies.
    WINDOW = 0.015

    # These will be used to restrict ourselves to
    local_windows = prices_df * WINDOW

    while True:
        # Grab the details about the first few legs that the workers further up
        # the pipeline have made for us.
        try:
            (source_df,
             leg1_bid_s,
             leg1_ask_s,
             leg2_bid_df,
             leg2_ask_df,
             base_open_margins,
             base_open_credits,
             base_close_credits) = input_q.get(timeout=0.1)
        except queue.Empty:
            # There's nothing in the queue. Check the number of workers up the
            # pipeline are still working and stop working if they are done.
            if prev_counter.value == 0:
                break
            else:
                # There are still source processes working, so keep trying to
                # get data.
                continue

        # This value is the same for all of the rows in the base DataFrame we
        # received.
        leg1_strike = source_df.loc[0, 'leg1_strike']

        # We'll be working on both short and long butterflies
        source_dfs = {s: source_df.copy() for s in ('short', 'long')}

        for side in source_dfs.keys():
            mid_leg_num = '1' if side == 'short' else '2'
            leg_name = 'leg{}'.format(mid_leg_num)

            # Duplicate all middle leg to be be used for leg 3 as well.
            for c in filter(lambda c: leg_name in c, source_df.columns):
                source_dfs[side].insert(
                    0,
                    c.replace(mid_leg_num, '3'),
                    source_df[c]
                )

            # The remainder of the code is going to need to get the details on
            # legs 1 through 3 for specific timepoints. Facilitate this by using
            # setting the index of the source DataFrames.
            source_dfs[side].set_index(
                ['open_time', 'leg1_strike', 'leg2_strike'],
                inplace=True
            )

        # Walk through each of the (leg 1, leg 2) combinations, treating
        # them as we did leg 1 in the vertical spread worker. That is, take legs
        # 1 through 3 as constant and work with all of the possible leg 4s.
        for leg2_strike, open_margins_12 in base_open_margins.items():

            leg2_ask_s = leg2_ask_df[leg2_strike]
            leg2_bid_s = leg2_bid_df[leg2_strike]

            count_msg = 'count({},{},:) ='.format(
                int(leg1_strike), int(leg2_strike))

            # We need to figure out if leg 1 is higher or lower than leg 2 and
            # then we want to find all of the strikes which would complete the
            # butterfly (i.e., those more expensive thatn the middle strike).
            long_bfly = leg2_strike > leg1_strike
            if long_bfly:
                middle_strike = leg2_strike
            else:
                middle_strike = leg1_strike

            leg4_strikes = all_strikes[all_strikes > middle_strike]

            if len(leg4_strikes) == 0:
                log('{} 0'.format(count_msg), long_bfly)
                continue

            # We only want to look at times where the middle strike was at least
            # somewhat close to the price of the underlying security.
            low_enough = (-local_windows + middle_strike) <= prices_df
            high_enough = (local_windows + middle_strike) >= prices_df

            valid_opens = prices_df[low_enough & high_enough].index

            if len(valid_opens) == 0:
                log('{} 0'.format(count_msg), long_bfly)
                continue

            leg4_ask_df = ask_df[leg4_strikes]
            leg4_bid_df = bid_df[leg4_strikes]

            if long_bfly:
                midleg_margin = leg2_bid_s
                leg4_open_margin = leg4_ask_df
            else:
                midleg_margin = leg1_ask_s
                leg4_open_margin = leg4_bid_df

            # Double the margin that the middle leg that contributes to the
            # trade then add to the leg 4 margin to get the total margin for the
            # butterfly.
            open_margins = leg4_open_margin.add(
                open_margins_12 + midleg_margin,
                axis='rows'
            )

            # Pretend that the too-expensive trades don't exist
            if max_margin is not None:
                open_margins[open_margins > max_margin] = np.nan

            base_df = open_margins.stack(
                level=0,
                dropna=False
            ).to_frame(name='open_margin')

            if base_open_credits is not None:

                if long_bfly:
                    # Selling leg 2 again, so we get a positive credit from
                    # people bidding for the option.
                    midleg_open_credit = leg2_bid_s
                    midleg_close_credit = -leg2_ask_s.fillna(0.01)
                    # Buying leg 4, so we pay a debit to the people who are
                    # asking for a certain amount for the option.
                    leg4_open_credit = -leg4_ask_df
                    leg4_close_credit = leg4_bid_df.fillna(0)
                else:
                    # Buying leg 1 again, so we pay a debit to the sellers
                    midleg_open_credit = -leg1_ask_s
                    midleg_close_credit = leg1_bid_s.fillna(0)
                    # Selling leg 4, so we get a credit from the buyers
                    leg4_open_credit = leg4_bid_df
                    leg4_close_credit = -leg4_ask_df.fillna(0.01)

                # Add the credits received from leg 2 once more and then
                # subtract that value from the credits received from each of the
                # potential leg 4s.
                open_credits = leg4_open_credit.add(
                    base_open_credits[leg2_strike] + midleg_open_credit,
                    axis='rows'
                )

                # Subtract the credits we'd need to pay to close out leg 2 once
                # more and then add the result to the credits we receive from
                # closing out the fourth leg.
                close_credits = leg4_close_credit.add(
                    base_close_credits[leg2_strike] + midleg_close_credit,
                    axis='rows'
                )

                # Get the maximum profit less fees
                max_profits = open_credits \
                              - trade_fees \
                              + utils.forward_looking_maximum(close_credits)

                # Convert the DataFrame into a Series which with the same
                # indexes as `open_margins`.
                max_profits = max_profits.stack(level=0, dropna=False)

                # Insert the maximum profit Series to the update DataFrame.
                base_df.insert(1, 'max_profit', max_profits)

            # Only look at the opens that are within a certain window and of
            # those trades, only look at the ones that we were able to open at
            # a reasonable margin.
            base_df = base_df.loc[valid_opens].dropna()

            # If there's nothing left after all that, we can just skip this
            # (leg 1, leg 2) combination.
            total_trades = base_df.shape[0]
            if total_trades == 0:
                log('{} 0'.format(count_msg), long_bfly)
                continue

            # Bring all of the indices into the dataframe to be columns
            base_df.reset_index(level=[0, 1], inplace=True)

            # Update the column names to be more descriptive.
            rename_dict = {
                'datetime': 'open_time',
                'strike': 'leg4_strike',
            }
            base_df = base_df.rename(columns=rename_dict)

            base_df.insert(0, 'leg1_strike', leg1_strike)
            base_df.insert(0, 'leg2_strike', leg2_strike)

            # Locate the rows in the source DataFrame that correspond to the
            # combination of leg 1 (static), leg 2 (static) and open time (
            # variable).
            rows_to_get = base_df[
                ['open_time','leg1_strike', 'leg2_strike']
            ].values

            # Make sure to use the DataFrame associated with our directionality
            # (i.e., short or long).
            to_concat = source_dfs['long' if long_bfly else 'short'].loc[
                [tuple(x) for x in rows_to_get]
            ].reset_index(drop=True)

            # Concat these rows into our data.
            base_df = pd.concat((base_df, to_concat),axis=1)

            # We still need to fill in the metadata for leg 4, so do that now.
            leg4_meta = option_type_df.loc[
                zip(base_df.open_time, base_df.leg4_strike)
            ].reset_index(drop=True)

            # There are the credits for the opens. If we're longing the
            # butterfly, we don't need to see what people are bidding for leg 4.
            # Additionally, the columns we just added via the concat() above
            # already contain the stock price, so we drop that as well.
            leg4_meta.drop(
                ['bidPrice' if long_bfly else 'askPrice', 'stock_price'],
                axis=1,
                inplace=True
            )

            # Any models will need to know if we are buying or selling this
            # leg, so add in this column (BUY = 1, SELL = -1).
            leg4_meta.insert(
                0,
                'action',
                1 if long_bfly else -1
            )

            # Rename the price column to "credits".
            leg4_meta.rename(
                columns={
                    ('askPrice' if long_bfly else 'bidPrice'): 'credit'
                },
                inplace=True
            )

            # We need to rename each of the columns
            leg4_meta.rename(
                columns={k: 'leg4_' + k for k in leg4_meta.keys()},
                inplace=True
            )

            log('{} {}'.format(count_msg, total_trades), long_bfly)

            output_q.put(
                pd.concat((base_df, leg4_meta), axis=1)
            )

    log('COMPLETE')

    # Signal that we are done
    with my_counter.get_lock():
        my_counter.value -= 1

def postprocessing_worker(
    id,
    input_q,
    output_q,
    prev_counter,
    my_counter,
    expiry_dt,
    dataframe_save_threshold,
    option_type,
    verbose=False,
):
    def log(message):
        if not verbose: return
        logging.debug(
            '{hdr} {msg}'.format(
                hdr=HEADER_TEMPLATE.format(name='SAVER', id=id),
                msg = message
            ))

    def save_piece(strategy_name, df_to_save):

        log('processing {} "{}" spreads'.format(
                df_to_save.shape[0], strategy_name))

        df_to_save.insert(0, 'expiry', expiry_dt)

        output_q.put((strategy_name, df_to_save))

    def describer(row):
        otype = 'call' if row.leg1_type == 'C' else 'put'
        if isinstance(row.leg4_type, str):
            strat = '{} butterfly'.format(otype)
            # But are we longing it or shorting it?
            if row.leg2_strike == row.leg3_strike:
                strat = 'long {}'.format(strat)
            else:
                strat = 'short {}'.format(strat)
        else:
            # leg 1 is the buy leg, leg 2 is the second leg
            # buy < sell := "bull"
            direction = 'bull' if row.leg1_strike < row.leg2_strike else 'bear'
            strat = 'vertical {} {}'.format(direction, otype)
        return strat

    log('START')

    strategy_spreads = {}
    while True:
        # Grab a strike for the leg we will be longing.
        try:
            source_df = input_q.get(timeout=0.1)
        except queue.Empty:
            # There's nothing in the queue. Check the number of workers up the
            # pipeline are still working and stop working if they are done.
            if prev_counter.value == 0:
                break
            else:
                # There are still source processes working, so keep trying to
                # get data.
                continue

        for i in range(1, config.TOTAL_LEGS + 1):
            source_df.insert(
                0,
                'leg{}_type'.format(i),
                option_type if source_df['leg{}_strike'.format(i)].iloc[0] != 0
                            else np.nan
            )

        # Add descriptions to the rows
        source_df.insert(0, 'description', source_df.apply(describer, axis=1))

        # Separate the source_df into DataFrames for each strategy
        for d in source_df.description.unique():
            strategy_spreads[d] = pd.concat((
                strategy_spreads.get(d, None),
                source_df[source_df.description == d]
            ))

        for strategy, df in strategy_spreads.items():
            if df is None or df.shape[0] < dataframe_save_threshold:
                continue

            # The buffer for this strategy has hit critical mass, so dump it
            # into a new file
            save_piece(strategy, df)

            # Reset the values and start filling up the buffer again
            strategy_spreads[strategy] = None

    # We're quitting, but we still need to check if there's anything in the
    # buffer to dump before quitting
    for strategy, df in strategy_spreads.items():
        if df is None:
            continue

        # The buffer for this strategy has hit critical mass, so dump it
        # into a new file
        save_piece(strategy, df)

    log('COMPLETE')

    # Signal that we are done
    with my_counter.get_lock():
        my_counter.value -= 1

def collect_spreads(
    options_df,
    process_counts=[1,4,3],
    max_margin=config.MARGIN,
    get_max_profit=True,
    dataframe_save_threshold=25000,
    verbose=0,
):

    def log(message):
        if verbose < 1:
            return
        logging.debug(
            '{hdr} {msg}'.format(
                hdr=HEADER_TEMPLATE.format(name='Main', id=' '),
                msg=message)
        )

    # We need several Queues to maximize efficiency, namely:

    # A Queue of the strikes to be used for the buying leg in the vertical
    # spreads worker.
    strikes_q = Queue()

    # A Queue between the vertical spreads worker and the butterfly worker.
    butterfly_q = Queue()

    # A Queue for the spreads workers to output their results for
    # postprocessing.
    processing_q = Queue()

    # A Queue between the post-processing worker and the main thread to be used
    # for yielding the complete results.
    output_q = Queue()

    # In addition to Queues, each worker in the pipeline needs to know when to
    # stop working. This is achieved by keeping a count of each worker and
    # letting each part of the pipeline check to see if the workers before them
    # in the pipeline are finished. For this, we need three Values, namely:

    # A Value for the count of vertical spread workers (vertical workers
    # decrement, butterfly workers check).
    vertical_counter = Value('i')

    # A Value for the count of butterfly spread workers (butterfly workers
    # decrement, postprocessing_workers check).
    butterfly_counter = Value('i')

    # A Value for the count of postprocessing workers (postprocessing workers
    # decrement, main thread checks).
    postprocessing_counter = Value('i')

    for expiry in sorted(options_df.index.unique(level=1)):
        logging.debug(expiry)

        expiry_df = options_df.xs(expiry, level=1)

        # Get the expiry as an aware datetime at the 4 pm closing bell
        expiry_dt = utils.expiry_string_to_aware_datetime(expiry)

        prices_df = expiry_df.groupby(level=[0])['stock_price'].first()

        for option_type in ('C', 'P'):
            log('Working on spreads based on {} options'.format(option_type))

            # Reset the thread counts.
            vertical_counter.value = process_counts[0]
            butterfly_counter.value = process_counts[1]
            postprocessing_counter.value = process_counts[2]

            # These three DataFrames are used over and over by all of the
            # workers.
            option_type_df = expiry_df.xs(option_type, level=1)
            bid_df = option_type_df['bidPrice'].unstack(level=[1])
            ask_df = option_type_df['askPrice'].unstack(level=[1])

            # Load in the list of strikes so that the threads can pull
            # them out
            for s in ask_df.columns:
                strikes_q.put(s)

            processes = []
            for i in range(process_counts[0]):
                p = Process(
                    target=vertical_spread_worker,
                    args=(
                        i,
                        strikes_q,
                        butterfly_q,
                        processing_q,
                        vertical_counter,
                        option_type_df,
                        bid_df,
                        ask_df,
                        prices_df,
                        get_max_profit,
                        max_margin,
                        verbose-1,
                    )
                )
                p.start()
                processes.append(p)

            for i in range(process_counts[1]):
                p = Process(
                    target=butterfly_spread_worker,
                    args=(
                        i,
                        butterfly_q,
                        processing_q,
                        vertical_counter,
                        butterfly_counter,
                        option_type_df,
                        bid_df,
                        ask_df,
                        prices_df,
                        max_margin,
                        verbose-1,
                    )
                )
                p.start()
                processes.append(p)

            for i in range(process_counts[2]):
                p = Process(
                    target=postprocessing_worker,
                    args=(
                        i,
                        processing_q,
                        output_q,
                        butterfly_counter,
                        postprocessing_counter,
                        expiry_dt,
                        dataframe_save_threshold,
                        option_type,
                        verbose-1,
                    )
                )
                p.start()
                processes.append(p)

            # Yield DataFrames whenever they become available
            while True:
                try:
                    label, df = output_q.get(timeout=0.1)
                except queue.Empty:
                    # We didn't receive anything. If the postprocessing workers
                    # are all finished, then we know that we've collected all
                    # the spreads we can.
                    if postprocessing_counter.value == 0:
                        break
                    else:
                        # There are still postprocessers working, so keep trying
                        # to get data.
                        continue

                # Give the calling process what it wants
                log('yielding')
                yield (label, df)

            # No more data, so join the processes and move on to something else.
            for p in processes:
                p.join()
