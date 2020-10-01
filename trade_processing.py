from datetime import datetime, timedelta, timezone
import multiprocessing
import numpy as np
import os
import pandas as pd
import pytz
import queue
import tempfile
import uuid

import config
import utils

DONE = 'finished'

HEADER_TEMPLATE = '{name:<15} {id:>2}:'

def collect_TA(ticker, dates):
    # return the technical analysis portion for the ticker to be used in the
    # final input vector
    pass

# ============================== Vertical spreads ==============================

def call_put_spread_worker(
    id,
    option_type_df,
    bid_df,
    ask_df,
    prices_df,
    strikes_q,
    output_q,
    get_max_profit=True,
    max_margin=None,
    verbose=False
):
    def log(message):
        if not verbose: return
        print(
            '{hdr} {msg}'.format(
                hdr=HEADER_TEMPLATE.format(name='CP_COLLECTOR', id=id),
                msg = message
            ))

    log('START')

    while True:
        # Grab a strike for the leg we will be longing
        try:
            leg1_strike = strikes_q.get(timeout=1)
        except queue.Empty:
            break

        leg2_strikes = bid_df.columns[bid_df.columns != leg1_strike]

        leg1_asks = ask_df[leg1_strike]

        # The second leg cannot include the first leg strike
        leg2_bid_df = bid_df[leg2_strikes]
        leg2_ask_df = ask_df[leg2_strikes]

        # Figure out how much margin we'd need for this trade
        open_margins = leg2_bid_df.add(leg1_asks, axis='rows')

        # Pretend that the too-expensive trades don't exist
        if max_margin is not None:
            open_margins[open_margins > max_margin] = np.nan
        open_margins = open_margins.stack(level=0, dropna=False)

        if get_max_profit:
            # Do some magic to get the maximum profits. Open credits is simple:
            # subtract the amount we pay for the first leg from the amount we
            # receive from the second.
            # NOTE: leave NaN in place to symbolize that this is not a viable
            #       trade. These trades will be removed at the end.
            open_credits = leg2_bid_df.sub(leg1_asks, axis='rows')

            # When looking at the close credits, make sure to replace NaN with 0
            # on the close sides, because NaN implies the trade was worthless.
            # This is good for leg 2 (since we buy it back for nothing at this
            # timepoint) and bad for leg 1 (since we can't sell it at this time
            # point).
            # NOTE: make this 0.01 for the leg 2 side since if we want to close
            #       early, we'd actually need to sell it for something.
            close_credits = (-leg2_ask_df.fillna(0.01)).add(
                                bid_df[leg1_strike].fillna(0), axis='rows')

            # Get the forward-looking maximum  profitby reversing the values,
            # applying a cumulative maximum, then flipping the result
            #       1,   0,   2,   1,   5,   0
            #       0,   5,   1,   2,   0,   1   (flip)
            #       0,   5,   5,   5,   5,   5   (cummax)
            #       5,   5,   5,   5,   5,   0   (flip)
            close_credits = pd.DataFrame(
                data    = np.flip(
                                np.maximum.accumulate(
                                    np.flip(close_credits.values))),
                columns = close_credits.columns,
                index   = close_credits.index
            )

            max_profits = open_credits + close_credits
            max_profits = max_profits.stack(level=0, dropna=False)
            leg1_df = pd.concat((open_margins, max_profits), axis=1)
        else:
            # Looks like the strikes, open times, and margins are all we need
            leg1_df = open_margins.to_frame()

        # Since we left the NaNs in place for the open credits, we can now strip
        # out the trades that we can't actually open because of:
        #   credits -> one or both sides weren't there to open, or
        #   margin  -> the combination was too pricey
        leg1_df.dropna(inplace=True)

        # If there's nothing left, then we can just skip this leg 1 strike
        total_trades = leg1_df.shape[0]
        if total_trades == 0:
            log('count({}) = 0'.format(int(leg1_strike)))
            continue

        # Bring all of the indices into the dataframe to be columns
        leg1_df.reset_index(level=[0, 1], inplace=True)

        # The column names are all funky after the reset_index
        rename_dict = {
            'datetime': 'open_time',
            'strike': 'leg2_strike',
            0: 'open_margin',
            1: 'max_profit'
        }
        leg1_df = leg1_df.rename(columns=rename_dict)

        # Add in the leg 1 strike
        leg1_strikes = np.full(total_trades, leg1_strike)
        leg1_df.insert(0, 'leg1_strike', leg1_strikes)

        # Also add in the strikes for (non-existent) legs 3 and 4
        leg1_df.insert(0, 'leg3_strike', np.zeros(total_trades))
        leg1_df.insert(0, 'leg4_strike', np.zeros(total_trades))

        all_open_times = leg1_df.open_time
        leg2_strikes = leg1_df.leg2_strike

        leg1_meta = option_type_df.loc[
            zip(all_open_times, leg1_strikes)].reset_index(drop=True)
        leg2_meta = option_type_df.loc[
            zip(all_open_times, leg2_strikes)].reset_index(drop=True)

        # Drop the bidPrice in the first leg and ask price in the second.
        leg1_meta.drop(['bidPrice'], axis=1, inplace=True)
        leg2_meta.drop(['askPrice'], axis=1, inplace=True)

        # Flip the sign of the askPrice for the first leg
        leg1_meta.askPrice *= -1

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

        output_q.put(
            pd.concat(
                (leg1_df.copy(), leg1_meta, leg2_meta, leg3_meta, leg4_meta),
                axis=1))

    log('COMPLETE')

    # Signal to one of the filesystem workers that one of the spread workers is
    # done
    output_q.put(DONE)

# The percent of the current price
WINDOW = 0.015

def butterfly_spread_worker(
    id,
    option_type_df,
    bid_df,
    ask_df,
    prices_df,
    strikes_q,
    output_q,
    get_max_profit=True,
    max_margin=None,
    verbose=False
):
    def log(message):
        if not verbose: return
        print(
            '{hdr} {msg}'.format(
                hdr=HEADER_TEMPLATE.format(name='BFLY_COLLECTOR', id=id),
                msg = message
            ))

    log('START')

    all_strikes = ask_df.columns

    while True:
        # Grab a strike for the leg we will be longing
        try:
            B_strike = strikes_q.get(timeout=1)
        except queue.Empty:
            break

        # Only look at time where this strike was at least somewhat close to the
        # price of the underlying security. Remove this to collect even the
        # weird trades
        viable_times = prices_df[
            (((B_strike - prices_df*WINDOW) < prices_df) &
               (prices_df < (B_strike + prices_df*WINDOW) ))[0]
        ].index

        if viable_times.shape[0] == 0:
            log('count(:,{},:) = 0 (outside range)'.format(int(B_strike)))
            continue

        # Ok, we know that there were at least some times where this trade made
        # sense according to current conventions. Let's use those times moving
        # forward
        viable_opens = bid_df.index.isin(viable_times)
        viable_bid_df = bid_df[viable_opens]
        viable_ask_df = ask_df[viable_opens]

        # The lower strike
        A_strikes = all_strikes[all_strikes < B_strike]

        # The higher strike
        C_strikes = all_strikes[all_strikes > B_strike]

        if 0 in (len(A_strikes), len(C_strikes)):
            log('count(:,{},:) = 0'.format(int(B_strike)))
            continue

        B_bids = viable_bid_df[B_strike]
        C_asks = viable_ask_df[C_strikes]

        for A_strike in A_strikes:
            A_asks = viable_ask_df[A_strike]

            # Figure out how much margin we'd need for this trade
            open_margins = C_asks.add(2 * B_bids + A_asks, axis='rows')

            # Pretend that the too-expensive trades don't exist
            if max_margin is not None:
                open_margins[open_margins > max_margin] = np.nan

            # Fold this out so that we have a new index (leg 4 strikes)
            open_margins = open_margins.stack(level=0, dropna=False)

            if get_max_profit:
                # Do some magic to get the maximum profits.

                # Open credits is simple:
                # subtract the amount we pay for legs 1 (A) and leg 4 (C) from
                # the amount we receive from legs 2 and 3 (both B).
                # NOTE: leave NaN in place to symbolize that this is not a
                #       viable trade. These trades will be removed at the end.
                open_credits = (-C_asks).add(2 * B_bids - A_asks, axis='rows')

                # When looking at the close credits, make sure to replace NaN
                # with 0 on the close sides, because NaN implies the trade was
                # worthless. This is good for legs 2 and 3 (since we buy them
                # back for nothing at this timepoint) and bad for leg 1 and leg
                # 4 (since we can't sell them this time point).
                # NOTE: make this 0.01 for leg 2 and leg 3 since if we want to
                #       close early, we'd actually need to sell it for
                #       something.
                close_credits = bid_df[C_strikes].fillna(0).add(
                    (bid_df[A_strike].fillna(0)
                    - 2*ask_df[B_strike].fillna(0.01)),
                    axis='rows')

                # Get the forward-looking maximum  profitby reversing the
                # values, applying a cumulative maximum, then flipping the
                # result.
                #       1,   0,   2,   1,   5,   0
                #       0,   5,   1,   2,   0,   1   (flip)
                #       0,   5,   5,   5,   5,   5   (cummax)
                #       5,   5,   5,   5,   5,   0   (flip)
                # Trim down the close_credits so that we only look at the time
                # indices that match our viable_opens
                close_credits = pd.DataFrame(
                    data    = np.flip(
                                    np.maximum.accumulate(
                                        np.flip(close_credits.values))),
                    columns = close_credits.columns,
                    index   = close_credits.index
                )[viable_opens]

                max_profits = open_credits + close_credits
                max_profits = max_profits.stack(level=0, dropna=False)
                a2b_df = pd.concat((open_margins, max_profits), axis=1)

            else:
                # Looks like the strikes, open times, and margins are all we
                # need
                a2b_df = open_margins.to_frame()

            # Since we left the NaNs in place for the open credits, we can now
            # strip out the trades that we can't actually open because of:
            #   credits -> one or more sides weren't there to open, or
            #   margin  -> the combination was too pricey
            a2b_df.dropna(inplace=True)

            # If there's nothing left, then we can just skip this combination
            # of legs 1, 2 and 3
            total_trades = a2b_df.shape[0]
            if total_trades == 0:
                log('count({},{},:) = 0'.format(int(A_strike),int(B_strike)))
                continue

            # Bring all of the indices into the dataframe to be columns
            a2b_df.reset_index(level=[0, 1], inplace=True)

            # The column names are all funky after the reset_index
            rename_dict = {
                'datetime': 'open_time',
                'strike': 'leg4_strike',
                0: 'open_margin',
                1: 'max_profit'
            }
            a2b_df = a2b_df.rename(columns=rename_dict)

            # Add in the strikes for legs 1, 2 and 3
            leg1_strikes = np.full(total_trades, A_strike)
            leg2_strikes = np.full(total_trades, B_strike)
            a2b_df.insert(0, 'leg1_strike', leg1_strikes)
            a2b_df.insert(0, 'leg2_strike', leg2_strikes)
            a2b_df.insert(0, 'leg3_strike', leg2_strikes)

            all_open_times = a2b_df.open_time
            leg4_strikes = a2b_df.leg4_strike

            leg1_meta = option_type_df.loc[
                zip(all_open_times, leg1_strikes)].reset_index(drop=True)
            leg2_meta = option_type_df.loc[
                zip(all_open_times, leg2_strikes)].reset_index(drop=True)
            leg4_meta = option_type_df.loc[
                zip(all_open_times, leg4_strikes)].reset_index(drop=True)

            # There are the credits for the opens. Since we're buying legs 1 (A)
            # and 4 (C), we get rid of their bidPrices and we invert the
            # askPrice
            leg1_meta.drop(['bidPrice'], axis=1, inplace=True)
            leg4_meta.drop(['bidPrice'], axis=1, inplace=True)
            leg1_meta.askPrice *= -1
            leg4_meta.askPrice *= -1

            # Meanwhile, we're selling legs 2 and 3, so get rid of the askPrice
            leg2_meta.drop(['askPrice'], axis=1, inplace=True)

            # Rename the prices to "credits"
            leg1_meta.rename(columns={'askPrice': 'credit'}, inplace=True)
            leg4_meta.rename(columns={'askPrice': 'credit'}, inplace=True)
            leg2_meta.rename(columns={'bidPrice': 'credit'}, inplace=True)

            # Make a copy of leg2_meta to get the third leg
            leg3_meta = leg2_meta.copy()

            # We need to rename each of the columns
            leg1_meta.rename(columns={k: 'leg1_' + k for k in leg1_meta.keys()},
                             inplace=True)
            leg2_meta.rename(columns={k: 'leg2_' + k for k in leg2_meta.keys()},
                             inplace=True)
            leg3_meta.rename(columns={k: 'leg3_' + k for k in leg3_meta.keys()},
                             inplace=True)
            leg4_meta.rename(columns={k: 'leg4_' + k for k in leg4_meta.keys()},
                             inplace=True)

            log('count({},{},:) = {}'.format(
                    int(A_strike), int(B_strike), total_trades))

            output_q.put(
                pd.concat(
                    (a2b_df.copy(), leg1_meta, leg2_meta, leg3_meta, leg4_meta),
                    axis=1))

    log('COMPLETE')

    # Signal to one of the filesystem workers that one of the spread workers is
    # done
    output_q.put(DONE)

def filesystem_worker(
    id,
    input_q,
    working_dir,
    ticker,
    epoch_expiry,
    prices_df,
    max_spreads,
    option_type,
    get_max_profit=True,
    winning_profit=None,
    loss_win_ratio=None,
    ignore_loss=None,
    verbose=False,
):
    def log(message):
        if not verbose: return
        print(
            '{hdr} {msg}'.format(
                hdr=HEADER_TEMPLATE.format(name='SAVER', id=id),
                msg = message
            ))

    def save_piece(strategy_name, df_to_save):

        log('processing {} "{}" spreads'.format(
                df_to_save.shape[0], strategy_name))

        if get_max_profit:
            if winning_profit is not None and loss_win_ratio is not None:
                # Determine the max profits when purchasing one of these trades
                losers = df_to_save[df_to_save.max_profit < winning_profit]
                winners = df_to_save[df_to_save.max_profit >= winning_profit]
                total_winners = winners.shape[0]

                # Get at most the desired ratio of losers to winners, using the
                # losers that were closest to profit
                losers_to_get = total_winners * loss_win_ratio

                df_to_save = pd.concat((
                    winners,
                    losers.sort_values(
                        by='max_profit', ascending=False)[:losers_to_get]
                ))
            if ignore_loss is not None:
                # Don't bother with trades we won't be using for training
                df_to_save = df_to_save[df_to_save[1] > ignore_loss]

        log('adding security prices')

        try:
            df_to_save['stock_price'] = prices_df.loc[
                df_to_save.open_time].values[:, 0]
        except KeyError:
            # It's likely that the server did not store some of the desired
            # times. Just eat the loss and get rid of these times
            to_remove = []
            opens = df_to_save.open_time
            indices = prices_df.index
            for i in range(opens.shape[0]):
                otime = opens.iloc[i]
                if otime not in indices and otime not in to_remove:
                    to_remove.append(otime)
            log(('Removing times that do not appear in prices DatFrame:'
                   '\n{}').format(to_remove))
            for otime in to_remove:
                df_to_save = df_to_save[df_to_save.open_time != otime]
            df_to_save['stock_price'] = prices_df.loc[
                df_to_save.open_time].values[:, 0]

        # Convert open_time to minutes_to_expiry, paying attention to timezones
        log('converting open time to minutes to expiry')

        # Get the opens as a UTC timedelta
        epoch_opens = df_to_save.open_time.apply(utils.get_epoch_timedelta)

        df_to_save.insert(
            0,
            'minutes_to_expiry',
            (epoch_expiry - epoch_opens).apply(lambda x: x.total_seconds())//60
        )

        # Make sure that the filename includes the strategy type so that we can
        # reference this without opening up the DataFrame
        filepath = os.path.join(
            working_dir,
            '{name}-{id}.bz2'.format(
                name=strategy_name.replace(' ', '_'),
                id=uuid.uuid4().hex
            )
        )
        log('saving to {}'.format(filepath))
        df_to_save.reset_index(drop=True).to_pickle(filepath)

    def describer(row):
        otype = 'call' if row.leg1_type == 1 else 'put'
        if row.leg4_type != 0:
            strat = '{} butterfly'.format(otype)
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
        item = input_q.get()

        # Quit in response to a spread worker quitting
        if isinstance(item, str) and item == DONE:
            log('Got {} signal from other side. Quitting.'.format(DONE))
            break

        # Add in a column showing which option type was in use for each leg.
        # Use the values of 1 and -1 to that 0 can be used to signify an empty
        # leg when working with the model
        type_array = np.ones(item.shape[0])
        empty_array = np.zeros(item.shape[0])
        for i in range(1, config.TOTAL_LEGS + 1):
            if item['leg{}_strike'.format(i)].iloc[0] != 0:
                leg_array = (-1 if option_type == 'P' else 1) * type_array
            else:
                leg_array = empty_array
            item.insert(0, 'leg{}_type'.format(i), leg_array)

        # Add descriptions to the rows
        item.insert(0, 'description', item.apply(describer, axis=1))

        # Separate the item into DataFrames for each strategy
        for d in item.description.unique():
            strategy_spreads[d] = pd.concat((
                strategy_spreads.get(d, None), item[item.description == d]))

        for strategy, df in strategy_spreads.items():
            if df is None or df.shape[0] < max_spreads:
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

def collect_spreads(
    ticker,
    expiry,
    options_df,
    prices_df=None,
    procs_pairs=5,
    max_margin=config.MARGIN,
    ignore_loss=None,
    get_max_profit=True,
    winning_profit=None,
    loss_win_ratio=None,
    max_spreads_per_file=25000,
    verbose=False,
    debug=False
):
    # Make sure that there is an output directory ready for all of the data.
    # This directory will be returned when done
    tmpdir = tempfile.mkdtemp(prefix='tp-')

    # First we need a Queue of the strikes to be used for the buying leg.
    strikes_q = multiprocessing.Queue()

    # And the threads will put their resulting DataFrames into this queue
    working_q = multiprocessing.Queue()

    # Get the expiry as a UTC timedelta from epoch to the 4 pm closing bell
    epoch_expiry = utils.expiry_string_to_epoch_delta(expiry)

    # Get the full set of prices that occured during this dataframe
    if prices_df is None:
        if verbose:
            print('Collecting security prices')
        all_times = options_df.index.get_level_values(level=0)
        prices_df = utils.get_security_prices(
            ticker    = ticker,
            start_dt  = all_times[0]-timedelta(minutes=5),
            end_dt    = all_times[-1],
            frequency = 'FiveMinutes'
        )

    for o in ('C', 'P'):
        if verbose:
            print('Working on spreads based on ' + o)

        # These three DataFrames are used over and over by all of the workers.
        option_type_df = options_df.xs(o, level=1)
        bid_df = option_type_df['bidPrice'].unstack(level=[1])
        ask_df = option_type_df['askPrice'].unstack(level=[1])

        if not debug:
            for collecter in (call_put_spread_worker, butterfly_spread_worker):
                # Load in the list of strikes so that the threads can pull them
                # out
                for s in ask_df.columns:
                    strikes_q.put(s)

                processes = []
                for i in range(procs_pairs):
                    p = multiprocessing.Process(
                        target=collecter,
                        args=(i,
                              option_type_df,
                              bid_df,
                              ask_df,
                              prices_df,
                              strikes_q,
                              working_q,
                              get_max_profit,
                              max_margin,
                              verbose,)
                    )
                    p.start()
                    processes.append(p)

                for i in range(procs_pairs):
                    p = multiprocessing.Process(
                        target=filesystem_worker,
                        args=(i,
                              working_q,
                              tmpdir,
                              ticker,
                              epoch_expiry,
                              prices_df,
                              max_spreads_per_file,
                              o,
                              get_max_profit,
                              winning_profit,
                              loss_win_ratio,
                              ignore_loss,
                              verbose,)
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

        else:
            call_put_spread_worker(
                0,
                option_type_df,
                bid_df,
                ask_df,
                strikes_q,
                working_q,
                max_margin,
                verbose,
            )

    return tmpdir
