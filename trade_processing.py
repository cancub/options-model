from datetime import datetime, timedelta, timezone
import multiprocessing
import numpy as np
import os
import pandas as pd
import pytz
import queue
import tempfile
import uuid

from questrade_helpers import QuestradeSecurities

import config

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
    buy_strikes_q,
    output_q,
    max_margin=None,
    verbose=False
):
    hdr = 'COLLECTOR {:>2}:'.format(id)
    if verbose:
        print('{} START'.format(hdr))

    while True:
        # Grab a strike for the leg we will be longing
        try:
            leg1_strike = buy_strikes_q.get(timeout=1)
        except queue.Empty:
            break

        leg1_asks = ask_df[leg1_strike]

        # Figure out how much margin we'd need for this trade
        open_margins = bid_df.add(leg1_asks, axis='rows')

        # Pretend that the too-expensive trades don't exist
        if max_margin is not None:
            open_margins[open_margins > max_margin] = np.nan
        open_margins = open_margins.stack(level=0, dropna=False)

        # Do some magic to get the maximum profits. Open credits is simple:
        # subtract the amount we pay for the first leg from the amount we
        # receive from the second.
        # NOTE: leave NaN in place to symbolize that this is not a viable trade.
        #       these trades will be removed at the end.
        open_credits = bid_df.sub(leg1_asks, axis='rows')

        # When looking at the close credits, make sure to replace NaN with 0 on
        # the close sides, because NaN implies the trade was worthless. This is
        # good for leg 2 (since we buy it back for nothing at this timepoint)
        # and bad for leg 1 (since we can't sell it at this time point).
        # NOTE: make this 0.01 for the leg 2 side since if we want to close
        #       early, we'd actually need to sell it for something.
        close_credits = (-bid_df.fillna(0.01)).add(
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

        # Since we left the NaNs in place for the open credits, we can now strip
        # out the trades that we can't actually open because of:
        #   credits -> one or both sides weren't there to open, or
        #   margin  -> the combination was too pricey
        leg1_df.dropna(inplace=True)

        # If there's nothing left, then we can just skip this leg 1 strike
        total_trades = leg1_df.shape[0]
        if total_trades == 0:
            if verbose:
                print('{} count({}) = 0'.format(hdr, int(leg1_strike)))
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

        # We need to rename each of the columns
        leg1_meta.rename(
            columns={k: 'leg1_' + k for k in leg1_meta.keys()}, inplace=True)
        leg2_meta.rename(
            columns={k: 'leg2_' + k for k in leg2_meta.keys()}, inplace=True)

        if verbose:
            print('{} count({}) = {}'.format(
                hdr, int(leg1_strike), total_trades))

        output_q.put(pd.concat((leg1_df.copy(), leg1_meta, leg2_meta), axis=1))

    if verbose:
        print('{} COMPLETE'.format(hdr))

def filesystem_worker(
    id,
    input_q,
    working_dir,
    ticker,
    epoch,
    epoch_expiry,
    prices_df,
    max_spreads,
    option_type,
    winning_profit=None,
    loss_win_ratio=None,
    ignore_loss=None,
    verbose=False,
):
    hdr = 'SAVER {:>2}:'.format(id)
    spread_list      = []
    def save_piece():
        # Get ready to save by building the start of the DataFrame
        df_to_save = pd.concat(spread_list)

        if winning_profit is not None and loss_win_ratio is not None:
            # Determine the max profits when purchasing one of these trades
            losers = df_to_save[df_to_save.max_profit < winning_profit]
            winners = df_to_save[df_to_save.max_profit >= winning_profit]
            total_winners = winners.shape[0]

            # Get at most the desired ratio of losers to winners, using the losers that
            # were closest to profit
            losers_to_get = total_winners * loss_win_ratio

            df_to_save = pd.concat((
                winners,
                losers.sort_values(by='max_profit', ascending=False)[:losers_to_get]
            ))
        if ignore_loss is not None:
            # Don't bother with trades we won't be using for training
            df_to_save = df_to_save[df_to_save[1] > ignore_loss]

        # Update the count
        trades_in_memory = df_to_save.shape[0]

        if verbose:
            print('{} saving {} spreads'.format(hdr, trades_in_memory))
        # Add in a column showing which option type was in use for each leg.
        # Use the values of 1 and -1 to that 0 can be used to signify an empty
        # leg when working with the model
        type_array = np.ones(trades_in_memory)
        for leg in ['leg1', 'leg2']:
            df_to_save.insert(
                0,
                leg + '_type',
                (-1 if option_type == 'P' else 1) * type_array
            )

        if verbose:
            print('{} adding security prices'.format(hdr))

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
            print(('Removing times that do not appear in prices DatFrame:'
                   '\n{}').format(to_remove))
            for otime in to_remove:
                df_to_save = df_to_save[df_to_save.open_time != otime]
            df_to_save['stock_price'] = prices_df.loc[
                df_to_save.open_time].values[:, 0]

        # Convert open_time to minutes_to_expiry, paying attention to timezones
        if verbose:
            print('{} converting open time to minutes to expiry'.format(hdr))

        # Get the opens as a UTC timedelta
        epoch_opens = df_to_save.open_time.apply(
            lambda x: x.astimezone(timezone.utc) - epoch)

        df_to_save.drop('open_time', axis=1, inplace=True)
        df_to_save.insert(
            0,
            'minutes_to_expiry',
            (epoch_expiry - epoch_opens).apply(lambda x: x.total_seconds())//60
        )

        filepath = os.path.join(working_dir, '{}.bz2'.format(uuid.uuid4()))
        if verbose:
            print('{} saving to {}'.format(hdr, filepath))
        df_to_save.reset_index(drop=True).to_pickle(filepath)

    if verbose:
        print('{} START'.format(hdr))

    trades_in_memory = 0
    started = False
    while True:
        # Grab a strike for the leg we will be longing
        try:
            # Block until we get the very first element, after that, only wait
            # one second, max. This allows us to quickly finish up but not be
            # too impatient to start
            spread_df = input_q.get(timeout=4 if started else None)
            started = True
        except queue.Empty:
            break

        trades_in_memory += spread_df.shape[0]
        spread_list.append(spread_df)

        # Once we hit a critical mass of spreads, this triggers the generation
        # of a new file into which we dump all of the loaded spreads and then
        # continue
        if trades_in_memory < max_spreads:
            continue

        if verbose:
            print('{} processing {} spreads'.format(hdr, trades_in_memory))

        save_piece()

        # Reset the values and start filling up the buffer again
        spread_list = []
        trades_in_memory = 0

    if trades_in_memory > 0:
        save_piece()

    if verbose:
        print('{} COMPLETE'.format(hdr))

def collect_spreads(
    ticker,
    expiry,
    options_df,
    getter_procs=5,
    saver_procs=5,
    max_margin=config.MARGIN,
    ignore_loss=None,
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
    buy_strikes_q = multiprocessing.Queue()

    # And the threads will put their resulting DataFrames into this queue
    working_q = multiprocessing.Queue()

    # Get the expiry in seconds from epoch
    epoch = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    expiry_dt = datetime.strptime(expiry, '%Y-%m-%d') + timedelta(hours=16)

    # Add the time zone (is the next change in the fall or spring?)
    transitions = pytz.timezone('America/Toronto')._utc_transition_times
    dst = next(t for t in transitions if t > expiry_dt).month > 9
    expiry_dt = expiry_dt.replace(
        tzinfo=timezone(timedelta(hours=-4 if dst else -5)))

    # Get the epxiry as a UTC timedelta
    epoch_expiry = expiry_dt - epoch

    # Get the full set of prices that occured during this dataframe
    if verbose:
        print('Collecting security prices')
    all_times = options_df.index.get_level_values(level=0)
    qs = QuestradeSecurities()
    candles = qs.get_candlesticks(
        ticker,
        str(all_times[0]-timedelta(minutes=5)),
        str(all_times[-1]),
        'FiveMinutes'
    )
    prices_df = pd.DataFrame(
        data=[c['open'] for c in candles],
        index=pd.to_datetime([c['end'] for c in candles]),
    )

    for o in ('C', 'P'):
        if verbose:
            print('Working on spreads based on ' + o)

        # These three DataFrames are used over and over by all of the workers.
        option_type_df = options_df.xs(o, level=1)
        bid_df = option_type_df['bidPrice'].unstack(level=[1])
        ask_df = option_type_df['askPrice'].unstack(level=[1])

        # Load in the buy-leg strikes so that the threads can pull them out
        for s in ask_df.columns:
            buy_strikes_q.put(s)

        if not debug:
            processes = []
            for i in range(getter_procs):
                p = multiprocessing.Process(
                    target=call_put_spread_worker,
                    args=(i,
                          option_type_df,
                          bid_df,
                          ask_df,
                          buy_strikes_q,
                          working_q,
                          max_margin,
                          verbose,)
                )
                p.start()
                processes.append(p)

            for i in range(saver_procs):
                p = multiprocessing.Process(
                    target=filesystem_worker,
                    args=(i,
                          working_q,
                          tmpdir,
                          ticker,
                          epoch,
                          epoch_expiry,
                          prices_df,
                          max_spreads_per_file,
                          o,
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
                buy_strikes_q,
                working_q,
                max_margin,
                verbose,
            )

    return tmpdir
