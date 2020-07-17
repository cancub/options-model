from collections import OrderedDict
import datetime as dt
import multiprocessing
import numpy as np
import os
import pandas as pd
import queue
import re
import sys

# Entry and exit fees: 9.95 + 1/contract
#   ex: 1 contract : 9.95 + 1 = 10.95
#       2 contract : 9.95 + 2 = 11.95
# So for both entre and exit of one contracts
FEE = 2 * (9.95 + 1)

# The required profit that we want
MIN_PROFIT = 100

# The amount of money we're willing to spend to enter the trade
MAX_DEBIT = 500

# Used for questrade calculations
MARGIN = 6000

# The negative to positive ratio
NP_RATIO = 3

THREAD_COUNT = 10

def collect_TA(ticker, dates):
    # return the technical analysis portion for the ticker to be used in the
    # final input vector
    pass

# TODO:
# check how to find vega

def _get_basic_datetimes(times_strings):
    dt_match = re.compile(r'\d+-\d+-\d+ \d+:\d+')
    strptime_format = '%Y-%m-%d %H:%M'
    return np.array(list(map(
        lambda x: dt.datetime.strptime(
            re.match(dt_match, x).group(0), strptime_format),
        times_strings
    )))

def get_stock_prices(ticker, working_dir='pickles'):
    # Get the prices time series for the underlying security of this ticker
    all_stock_prices = pd.read_pickle(os.path.join(working_dir, 'price'))
    stock_dtimes = _get_basic_datetimes(all_stock_prices['datetime'])
    return all_stock_prices['ticker'].set_index(stock_dtimes, inplace=True)

def _process_options(data, metadata, dtimes):
    result = {}
    for otype in ['C', 'P']:
        # Filter metadata for this option type
        type_meta = metadata[metadata['type'] == otype]

        # Get the data (bid, ask) for these strikes
        bid_df = pd.DataFrame(
            data[0,:,list(type_meta.index)].T,
            index = dtimes,
            columns=type_meta['strike']
        )
        ask_df = pd.DataFrame(
            data[1,:,list(type_meta.index)].T,
            index = dtimes,
            columns=type_meta['strike']
        )

        # Get the DataFrames for all combinations of bid/ask + open/all
        result[otype] = {
            'strikes': type_meta['strike'], 'bid': bid_df, 'ask': ask_df}

    return result

# ================================= Single-Leg =================================

def collect_single_legs(ticker, working_dir='pickles'):
    working_dir = os.path.abspath(working_dir)
    expiries = sorted(os.listdir(working_dir))

    # We need to confirm that this is indeed an expiries home directory, which
    # must contain the prices of all stocks in a file called prices
    if 'price' not in expiries:
        raise TypeError(
            '{} is not a valid working directory'.format(working_dir))

    ticker = ticker.upper()

    # Use an ordered dictionary to maintain the order of the sorted list of
    # expiry dates
    result = OrderedDict()

    for exp in (e for e in expiries if e != 'price'):
        # If this expiry doesn't have info on our ticker, then it's not an
        # expiry that we want to associate with
        base_path = os.path.join(working_dir, exp, ticker)
        if not os.path.exists(base_path):
            continue

        # Collect all of the relevant info for this ticker and expiry
        data = np.load(base_path, allow_pickle=True)
        # In older versions of the metadata, indices do not correspond to the
        # option indices in the numpy array. Resetting them here fixes that and
        # does not affect newer metadata
        metadata = pd.read_pickle(base_path + '_meta').reset_index(drop=True)
        with open(base_path + '_times', 'r') as TF:
            dtimes = _get_basic_datetimes(TF.read().split('\n')[:-1])

        # Make sure all the quantities jive
        assert(data.shape[0] == 10)
        assert(data.shape[1] == len(dtimes))
        assert(data.shape[2] == len(metadata))

        exp_date = dt.datetime.strptime(exp, '%Y-%m-%d').date()

        # Alright, we now have all the information we need to do some
        # pre-processing with respect to viability
        result[exp_date] = _process_options(data, metadata, dtimes)

    return result

# ============================== Vertical spreads ==============================

def spread_worker(id, bid_df, ask_df, buy_strikes, profits, get_sell_strikes):
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
    trades_concat = []

    print('{:>2}: starting'.format(id))

    while True:
        # Grab a strike for the leg we will be buying
        try:
            buy_strike = buy_strikes.get()
        except queue.Empty:
            break

        # The strikes which we will sell (short)
        for sell_strike in get_sell_strikes(buy_strike):

            # Concat how much people are willing to pay for each leg at each
            # time
            # TODO: if bid doesn't exist, look at ask?
            market_values = pd.concat(
                (bid_df[buy_strike], bid_df[sell_strike]),
                axis=1
            )

            # Concat the credits (negative if buying) we will receive for
            # each leg
            open_credits = pd.concat(
                (-ask_df[buy_strike], bid_df[sell_strike]),
                axis=1
            )

            # The trade must have both elements existing at the same time
            # and, combined, their market value must not be over our margin
            # allowance.
            # NOTE: this is going to be the DataFrame we submit for the
            #       viable (but not necessarily profitable) trades for this
            #       pair of legs. We will be adding a few more columns if we
            #       for the closing part of the play
            open_credits = open_credits[
                (pd.notna(open_credits).all(axis=1)) &
                (market_values.sum(axis=1) * 100 < MARGIN)
            ]

            if len(open_credits) == 0:
                continue

            open_sum = open_credits.sum(axis=1)
            close_credits = pd.concat(
                (bid_df[buy_strike], -ask_df[sell_strike]),
                axis=1
            )

            # These are the lists that will be used to add new close-side
            # columns for the open_credits DataFrame
            close_profits = []
            close_times = []

            # Ok, now for each viable open time, figure out the maximum
            # profit if we were to:
            #   a) close at the same time
            #   b) TODO: close at different times
            for time_index in open_sum.index:
                open_profit = open_sum[time_index]
                # Get the closing trades after this open
                after_credits = close_credits[time_index:]

                # Of these closing trades, look at the ones that we can
                # close at the same time
                viable_close_credits = after_credits[pd.notna(
                    after_credits).all(axis=1)]

                # Check to see if there are even any times we can close this
                # trade
                if len(viable_close_credits) == 0:
                    # Remove this as a viable open and move on to the next
                    # time
                    open_credits.drop([time_index], inplace=True)
                    continue

                viable_close_credits = viable_close_credits.sum(axis=1)

                # From whatever is left, find the maxmimum credit received
                close_profits.append(
                    viable_close_credits.max() + open_profit)
                close_times.append(viable_close_credits.idxmax())

            total_trades = len(close_profits)

            # No viable trades for us with this leg combination
            if total_trades == 0:
                continue

            assert(total_trades == len(open_credits))

            # Add the new close columns to the DataFrame and stow it in the
            # list of dataframes we'll be concatenating
            open_credits['sell_leg'] = [sell_strike] * total_trades
            open_credits['profit'] = close_profits
            open_credits['close'] = close_times

            # The open dataframe still carries the strike number column names
            open_credits.rename(
                columns={
                    buy_strike: 'buy_leg_credit',
                    sell_strike: 'sell_leg_credit'
                },
                inplace=True
            )

            trades_concat.append(open_credits)

        # We've finished up all the possible trades with this buy leg. If there
        # were any valid trades, the buy-strike dataframe, add the strike price
        # column and return it to the main thread.
        total_trades = 0
        if len(trades_concat) > 0:
            trades_df = pd.concat(trades_concat)
            total_trades = len(trades_df)
            trades_df['buy_leg'] = [buy_strike] * total_trades
            profits.put(trades_df)

        print('{:>2}: count({}) = {}'.format(id, int(buy_strike), total_trades))

    print('{:>2}: done'.format(id))

def collect_spreads(ticker, bull_bear, put_call, working_dir='pickles',
                    vertical=True):
    # Error checking
    try:
        # Allow for bull, bear, BulL, BEAR, etc
        bull_bear = bull_bear.strip().lower()
    except AttributeError:
        raise TypeError(
            '`bull_bear` must be a string, not {}.'.format(
                put_call.__class__.__name__)
        )
    try:
        # Allow for P, C, put, call, PuT, cAll, etc
        PC_char = put_call.strip().upper()[0]
    except AttributeError:
        raise TypeError(
            '`put_call` must be a string, not {}.'.format(
                put_call.__class__.__name__)
        )
    except IndexError:
        raise ValueError('`put_call` cannot be an empty string.')

    if bull_bear not in ('bull', 'bear'):
        raise ValueError(
            '`bull_bear` must be in ["bull", "bear"] (case-insensitive)')
    if PC_char not in ('P', 'C'):
        raise ValueError(('`put_call` must be in ["p", "c", "put", "call"] '
                          '(case-insensitive)'))

    # Get all the necessary information to do some processing.
    single_legs = collect_single_legs(ticker, working_dir)

    # TODO: use to whittle things down to only those calls that are at or above
    # the money, as per the description
    # stock_prices = get_stock_prices(ticker, working_dir)

    # Ok, we got all of the data and metadata, so we're good to go. Let's build
    # the lambda we will use to get the sell strike from the buy strike and the
    # set of all strikes. Guido doesn't like multi-line lambdas, so we can't
    # make a lambda that generates a lambda AND use verbose variable names.
    if bull_bear == 'bull':
        # Buy LOWER strike call/put, sell HIGHER strike call/put
        def get_sell_strikes_gen(strikes):
            return lambda buy_strike: strikes[strikes > buy_strike]
    else:
        # Buy HIGHER strike call/put, sell LOWER strike call/put
        def get_sell_strikes_gen(strikes):
            return lambda buy_strike: strikes[strikes < buy_strike]

    result_concat = []

    # Build the thread-specific objects.

    # First we need a Queue of the strikes to be used for the buying leg.
    buy_strikes = multiprocessing.Queue()

    # And the threads will put their resulting DataFrames into this queue
    exp_profits = multiprocessing.Queue()

    for exp_date, exp_dict in single_legs.items():
        # Focus exclusively on puts since this is a put spread
        exp_info = exp_dict[PC_char]

        bid_df = exp_info['bid']
        ask_df = exp_info['ask']

        # Pull out the relevant info
        all_strikes = exp_info['strikes']

        # Load in the buy-leg strikes so that the threads can pull them out
        for s in all_strikes:
            buy_strikes.put(s)

        processes = []
        for i in range(THREAD_COUNT):
            p = multiprocessing.Process(
                target=spread_worker,
                args=(i, bid_df, ask_df, buy_strikes, exp_profits,
                      get_sell_strikes_gen(all_strikes),)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Did we even get any trades out of this expiry?
        if exp_profits.qsize() == 0:
            continue

        # Pull the DataFrames out of the Queue such that we can concat them into
        # one, expiry DataFrame and also empty the Queue for the next expiry
        concat_list = []
        while True:
            try:
                concat_list.append(exp_profits.get())
            except queue.Empty:
                break

        exp_df = pd.concat(concat_list)
        exp_df['expiry'] = [exp_date] * len(exp_df)

        result_concat.append(exp_df)

    return pd.concat(result_concat)

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

