import multiprocessing
import numpy as np
import os
import pandas as pd
import queue

import config
import utils

def collect_TA(ticker, dates):
    # return the technical analysis portion for the ticker to be used in the
    # final input vector
    pass

# TODO:
# check how to find vega

def get_stock_prices(ticker, working_dir='pickles'):
    # Get the prices time series for the underlying security of this ticker
    all_stock_prices = pd.read_pickle(os.path.join(working_dir, 'price'))
    stock_dtimes = utils.get_basic_datetimes(all_stock_prices['datetime'])
    return all_stock_prices['ticker'].set_index(stock_dtimes, inplace=True)

# ============================== Vertical spreads ==============================

def spread_worker(id, bid_df, ask_df, buy_strikes, profits, get_sell_strikes,
                  verbose=False):
    '''
    A thread-safe worker which takes care of collecting profits DataFrames for
    one of the standard bull/bear call/put spreads.

    The trick here is to provide only the bid_df and ask_df specific to the type
    of option (i.e., call or put) used by the spread.

    Most importantl though is the get_sell_strikes() function, which does the magic of
    differentiation between the types of spreads. This worker will be walking
    Through all of the strikes it can for the leg we will _buy_ and will then
    rely on get_sell_strikes() to return the list of strikes to be used for the leg we
    will _sell_.
    '''
    trades_data = {
        'open_time': [],
        'open_margin': [],
        'open_credit': [],
        'leg1_strike': [],
        'leg2_strike': [],
        'max_profit': [],
        'close_time': [],
    }

    while True:
        # Grab a strike for the leg we will be longing
        try:
            buy_strike = buy_strikes.get(timeout=1)
        except queue.Empty:
            break

        total_trades = 0
        long_bids = bid_df[buy_strike]

        # Get the strikes for the legs we will sell short in combinations with
        # the long leg
        all_sell_strikes = get_sell_strikes(buy_strike)

        if all_sell_strikes.size == 0 and verbose:
            print('{:>2}: count({}) = 0'.format(id, int(buy_strike)))
            continue

        short_bids = bid_df[all_sell_strikes]

        # Subtract the value we're paying to open the long leg of the trade
        open_credits = short_bids.sub(ask_df[buy_strike], axis='rows')

        # Set all of the too-expensive opens to NaN so that we can ignore them
        # Note that we need to fill all the NaN in this original bid DataFrames
        # with 0 so that we don't make the mistake of counting
        #   3 gagillion + NaN = NaN
        # as "OK"
        open_margin = short_bids.fillna(0).add(long_bids.fillna(0), axis='rows')
        open_credits[open_margin > config.MARGIN] = np.nan

        # Get rid of all timepoints that have no viable opens for any
        # combination
        open_credits.dropna(axis=0, how='all', inplace=True)
        viable_opens = open_credits.index

        if len(viable_opens) == 0 and verbose:
            print('{:>2}: count({}) = 0'.format(id, int(buy_strike)))
            continue

        # We know that there is at least one short strike that is viable. So now
        # we get rid of short strikes representing combinations that have no
        # viable opens
        open_credits.dropna(axis=1, how='all', inplace=True)
        viable_strikes = open_credits.columns

        # Get all the (negative) credits from buying options to close out the
        # short legs
        short_asks = -ask_df[viable_strikes]

        # These are the lists that will be used to add new close-side
        # columns for the open_credits DataFrame

        # Ok, now for each viable open time, figure out the maximum
        # profit if we were to:
        #   a) close at the same time
        #   b) TODO: close at different times
        for open_time in viable_opens:
            # Whittle the long option bids df down to the timepoints that we can
            # use for closing trades
            # NOTE: we can't close a trade right after we open it, hence
            #       skipping the first open time with .iloc[1:]
            close_long_bids = long_bids[open_time:].iloc[1:]
            if close_long_bids.shape[0] == 0:
                continue
            first_close_time = close_long_bids.index[0]

            # Figure out how much credit we would receive for closing out the
            # trades at each of the viable close times
            close_credits = short_asks.loc[first_close_time:].add(
                close_long_bids, axis='rows')

            # Take the maximum total close credits for each strike and add them
            # to theit respective total open credits
            max_profits = open_credits.loc[open_time].add(close_credits.max())
            # Get rid of the trades we weren't able to close out using this
            # opening time
            max_profits.dropna(inplace=True)
            max_profits_strikes = max_profits.index

            # Get the times associated with the above profits
            max_profit_closes = close_credits.idxmax()[max_profits_strikes]

            # Use only the elements from the open credits that correspond the to
            # strikes we were unable to use
            open_time_credits = open_credits.loc[open_time, max_profits_strikes]
            open_time_margin = open_margin.loc[open_time, max_profits_strikes]

            # Double check that we didn't mess this up earlier
            if open_time_margin.max() > config.MARGIN:
                raise Exception('Error with maximum open credit')

            total_trades_for_open = len(max_profits)

            total_trades += total_trades_for_open

            # Finally, add all of the data to the dict
            trades_data['open_time'] += [open_time] * total_trades_for_open
            trades_data['open_margin'] += open_time_margin.tolist()
            trades_data['open_credit'] += open_time_credits.tolist()
            trades_data['leg2_strike'] += max_profits_strikes.tolist()
            trades_data['max_profit'] += max_profits.tolist()
            trades_data['close_time'] += max_profit_closes.tolist()

            # Check that the lengths still match
            new_length = len(trades_data['open_time'])
            for k, v in trades_data.items():
                # We haven't added the long strikes yet
                if k != 'leg1_strike' and new_length != len(v):
                    import pdb; pdb.set_trace()

        # We've finished up all the possible trades with this buy leg. If there
        # were any valid trades, the buy-strike dataframe, add the strike price
        # column and return it to the main thread.
        if total_trades > 0:
            trades_data['leg1_strike'] += [-buy_strike] * total_trades

        if verbose:
            print('{:>2}: count({}) = {}'.format(
                id, int(buy_strike), total_trades))

    profits.put(pd.DataFrame(trades_data).set_index('open_time'))

    if verbose:
        print('{:>2}: done'.format(id))


def collect_spreads(
        ticker, expiry, directions=['bull', 'bear'],
        options=['c','p'], vertical=True, num_procs=10, verbose=False,
        debug=False):
    # Error checking
    if not isinstance(directions, list):
        raise TypeError(
            '`directions` must be a list. Got {}'.format(
                directions.__class__.__name__)
        )
    if not isinstance(options, list):
        raise TypeError(
            '`options` must be a list. Got {}'.format(
                options.__class__.__name__)
        )
    if len(directions) == 0:
        print('No directions specified. Exiting')
    if len(options) == 0:
        print('No options specified. Exiting')
    for i in range(len(directions)):
        try:
            # Allow for bull, bear, BulL, BEAR, etc
            directions[i] = directions[i].strip().lower()
        except AttributeError:
            raise TypeError(
                '`directions` elements must be strings, not {}.'.format(
                    directions[i].__class__.__name__)
            )
        if directions[i] not in ('bull', 'bear'):
            raise ValueError(
                ('`directions` elements must be in ["bull", "bear"] '
                 '(case-insensitive)')
            )
    for i in range(len(options)):
        try:
            # Allow for P, C, put, call, PuT, cAll, etc
            options[i] = options[i].strip().upper()[0]
        except AttributeError:
            raise TypeError(
                '`options` elements must be strings, not {}.'.format(
                    options[i].__class__.__name__)
            )
        except IndexError:
            raise ValueError('`options` elements cannot be an empty string.')
        if options[i] not in ('P', 'C'):
            raise ValueError(
                ('`options` elements must be in ["p", "c", "put", "call"] '
                 '(case-insensitive)')
            )

    # We're good to go. Start by building the thread-specific objects. We may
    # not be using threads, but even in this case the same thread objects can
    # be used.

    # First we need a Queue of the strikes to be used for the buying leg.
    buy_strikes = multiprocessing.Queue()

    # And the threads will put their resulting DataFrames into this queue
    profits = multiprocessing.Queue()

    tik_exp_df = utils.retrieve_options(ticker, expiry)

    result_df_list = []

    for o in options:

        # TODO: convert the strike into the column numbers?
        bid_df = tik_exp_df.xs(o, level=1)['bidPrice'].unstack(level=[1])
        ask_df = tik_exp_df.xs(o, level=1)['askPrice'].unstack(level=[1])

        for d in directions:

            if verbose:
                print(
                    'Working on a {} {} spread'.format(
                        d, 'call' if o == 'C' else 'put')
                )

            # Ok, we got all of the data and metadata, so we're good to go.
            # Let's build the lambda we will use to get the sell strike from the
            # buy strike and the set of all strikes. Guido doesn't like
            # multi-line lambdas, so we can't make a lambda that generates a
            # lambda AND use verbose variable names.
            if d == 'bull':
                # Buy LOWER strike call/put, sell HIGHER strike call/put
                def get_sell_strikes_gen(strikes):
                    return lambda buy_strike: strikes[strikes > buy_strike]
            else:
                # Buy HIGHER strike call/put, sell LOWER strike call/put
                def get_sell_strikes_gen(strikes):
                    return lambda buy_strike: strikes[strikes < buy_strike]

            # Pull out the relevant info
            all_strikes = bid_df.columns

            # Load in the buy-leg strikes so that the threads can pull them out
            for s in all_strikes:
                buy_strikes.put(s)

            spread_list = []
            if not debug:
                processes = []
                for i in range(num_procs):
                    p = multiprocessing.Process(
                        target=spread_worker,
                        args=(i, bid_df, ask_df, buy_strikes, profits,
                            get_sell_strikes_gen(all_strikes), verbose,)
                    )
                    p.start()
                    processes.append(p)

                # This is going to be a lot of buffered data in the Queue. The
                # poster on this topic https://stackoverflow.com/a/26738946
                # mentioned that we may need to read from the Queue *before* joining
                # Maybe the better solution is to reference a semaphor
                for _ in range(num_procs):
                    spread_list.append(profits.get())

                for p in processes:
                    p.join()

                spread_df = pd.concat(spread_list)
            else:
                spread_worker(0, bid_df, ask_df, buy_strikes, profits,
                            get_sell_strikes_gen(all_strikes), verbose)
                spread_df = profits.get()

            spread_df['direction'] = d
            spread_df['type'] = o

            result_df_list.append(spread_df)

    result_df = pd.concat(result_df_list)

    # Show the true values of the trade
    for k in ('open_margin', 'open_credit', 'max_profit'):
        result_df[k] *= 100

    return result_df

def bull_bear_phase_spread(ticker):
    '''
    A bull-bear phase spread can be constructed using near month call & put.
    '''
    pass



# =============================== Ratio Spreads ================================
'''
Our go to ratio-spread is a front-ratio spread. We normally do not route back-ratio spreads, which is where we are purchasing more options than we are selling, because this would be routed for a debit. We always prefer to collect premium and put ourselves in high probability situations. The beauty of this trade is that if we’re directionally wrong, it doesn’t matter if our spread expires OTM as long as we collect a credit - that will then be our profit. We route front-ratio spreads as a means to get into a long or short stock positon with a very beneficial breakeven point. We tend to use these strategies if we have a price target in mind for the underlying. We will usually place our short strike at that target, as that would yield max profit at expiration if the stock ends up there.
'''

def call_ratio_spread(ticker):
    '''
    A Call Front Ratio Spread is a neutral to bullish strategy that is created
    by purchasing a call debit spread with an additional short call at the short
    strike of the debit spread. The strategy is generally placed for a net
    credit so that there is no downside risk.

    Directional Assumption: Neutral to slightly bullish

    Setup:
    - Buy an ATM or OTM call option
    - Sell two further OTM call options at a higher strike

    Ideal Implied Volatility Environment : High

    Max Profit: Distance between long strike and short strike + credit received

    How to Calculate Breakeven(s): Short call strike + max profit potential
    '''
    pass

def put_ratio_spread(ticker):
    '''
    A Put Front Ratio Spread is a neutral to bearish strategy that is created by
    purchasing a put debit spread with an additional short put at the short
    strike of the debit spread. The strategy is generally placed for a net
    credit so that there is no upside risk.

    Directional Assumption: Neutral to slightly bearish

    Setup:
    - Buy an ATM or OTM put option
    - Sell two further OTM put options at a lower strike

    Ideal Implied Volatility Environment : High

    Max Profit: Distance between long strike and short strike + credit received

    How to Calculate Breakeven(s): Short put strike - max profit potential
    '''

# ================================ Backspreads =================================

def call_backspread(ticker):
    '''
    The call backspread (reverse call ratio spread) is a bullish strategy in
    options trading whereby the options trader writes a number of call options
    and buys more call options of the same underlying stock and expiration date
    but at a higher strike price. It is an unlimited profit, limited risk
    strategy that is used when the trader thinks that the price of the
    underlying stock will rise sharply in the near future.

    A 2:1 call backspread can be created by selling a number of calls at a lower
    strike price and buying twice the number of calls at a higher strike. 
    '''
    pass

def put_backspread(ticker):
    '''
    The put backspread is a strategy in options trading whereby the options
    trader writes a number of put options at a higher strike price (often
    at-the-money) and buys a greater number (often twice as many) of put options
    at a lower strike price (often out-of-the-money) of the same underlying
    stock and expiration date. Typically the strikes are selected such that the
    cost of the long puts is largely offset by the premium earned in writing the
    at-the-money puts. This strategy is generally considered very bearish but it
    can also serve as a neutral/bullish play under the right conditions.  
    '''
    pass

# ================================= convoluted =================================

def iron_condor(ticker):
    '''
    An Iron Condor is a directionally neutral, defined risk strategy that
    profits from a stock trading in a range through the expiration of the
    options. It benefits from the passage of time and any decreases in implied
    volatility.

    Directional Assumption: Neutral

    Setup:
    - Sell OTM Call Vertical Spread
    - Sell OTM Put Vertical Spread
    '''
    pass

def strangle(ticker):
    '''
    A short strangle is a position that is a neutral strategy that profits when the stock stays between the short strikes as time passes, as well as any decreases in implied volatility. The short strangle is an undefined risk option strategy.

    Directional Assumption: Neutral

    Setup:
    - Sell OTM Call
    - Sell OTM Put

    Ideal Implied Volatility Environment : High

    Max Profit: Credit received from opening trade

    How to Calculate Breakeven(s):
    - Downside: Subtract total credit from short put
    - Upside: Add total credit to short call 
    '''
    pass

def straddle(ticker):
    '''
    A short straddle is a position that is a neutral strategy that profits from
    the passage of time and any decreases in implied volatility. The short
    straddle is an undefined risk option strategy.

    Directional Assumption: Neutral

    Setup:
    - Sell ATM Call
    - Sell ATM Put

    Ideal Implied Volatility Environment : High

    Max Profit: Credit received from opening trade

    How to Calculate Breakeven(s):
    - Downside: Subtract initial credit from Put strike price
    - Upside: Add initial credit to the Call strike price
    '''
    pass

def butterfly_spread(ticker):
    '''
    A long butterfly spread is a neutral position that’s used when a trader
    believes that the price of an underlying is going to stay within a
    relatively tight range.

    Directional Assumption: Neutral

    Setup: This spread is typically created using a ratio of 1-2-1
    (1 ITM option, 2 ATM options, 1 OTM option).
    - Buy Call/Put (above short strike)
    - Sell 2 Calls/Puts
    - Buy Call/Put (below short strike)

    Ideal Implied Volatility Environment : High

    Max Profit:
    The distance between the short strike and long strike, less the debit paid.

    How to Calculate Breakeven(s):
    - Upside: Higher Long Option Strike - Debit Paid
    - Downside: Lower Long Option Strike + Debit Paid
    '''
    pass

def broken_wing_butterfly(ticker):
    '''
    A Broken Wing Butterfly is a long butterfly spread with long strikes that
    are not equidistant from the short strike. This leads to one side having
    greater risk than the other, which makes the trade slightly more directional
    than a standard long butterfly spread.

    Directional Assumption: Neutral / Slightly Directional

    Setup: Broken wing butterfly spreads can be constructed with either all
    calls or all puts. The trade is comprised of two short options and a long
    option above and below the short strike:
    - Buy Call/Put (above short strike)
    - Sell 2 Calls/Puts
    - Buy Call/Put (below short strike)
    '''
    pass

def calendar_spread(ticker):
    '''
    A Long Calendar Spread is a low-risk, directionally neutral strategy that
    profits from the passage of time and/or an increase in implied volatility.

    Directional Assumption: Neutral

    Setup: A calendar is comprised of a short option (call or put) in a
    near-term expiration cycle, and a long option (call or put) in a longer-term
    expiration cycle. Both options are of the same type and use the same strike
    price.

    - Sell near-term Put/Call
    - Buy longer-term Put/Call
    '''
    pass

def iron_fly(ticker):
    '''
    An Iron Fly is essentially an Iron Condor with call and put credit spreads
    that share the same short strike. This creates a very neutral position that
    profits from the passage of time and any decreases in implied volatility. An
    Iron Fly is synthetically the same as a long butterfly spread using the same
    strikes.

    Directional Assumption: Neutral

    Setup:
    - Buy OTM Put option
    - Sell Straddle (short call and short put at the same strike, typically
      At-The-Money)
    - Buy OTM Call option
    '''
    pass

def jade_lizard(ticker):
    '''
    A Jade Lizard is a slightly bullish strategy that combines a short put and a
    short call spread. The strategy is created to have no upside risk, which is
    done by collecting a total credit greater than the width of the short call
    spread.

    Directional Assumption: Neutral / Bullish

    Setup:
    - Sell OTM Put
    - Sell OTM Vertical Call Spread

    Ideal Implied Volatility Environment : High

    Max Profit: Credit received from opening trade. Max profit is realized when
    the stock price is between the short strikes at expiration.

    How to Calculate Breakeven(s):
    - Downside: Strike Price of short put - credit received 
    '''
    pass

