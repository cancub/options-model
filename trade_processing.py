from datetime import datetime, timedelta, timezone
import multiprocessing
import numpy as np
import os
import pandas as pd
import pytz
import queue

from questrade_helpers import QuestradeSecurities

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

def spread_worker(
    id,
    option_type_df,
    bid_df,
    ask_df,
    buy_strikes_q,
    profits_q,
    get_max_profit=False,
    max_margin=config.MARGIN,
    verbose=False
):
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

    result_dataframes = []

    while True:
        # Grab a strike for the leg we will be longing
        try:
            leg1_strike = buy_strikes_q.get(timeout=1)
        except queue.Empty:
            break

        leg1_asks = ask_df[leg1_strike]
        leg1_bids = bid_df[leg1_strike]

        # Figure out how much margin we'd need for this trade
        open_margins = bid_df.add(leg1_asks, axis='rows')

        # Pretend that the too-expensive trades don't exist
        open_margins[open_margins > max_margin] = np.nan
        open_margins = open_margins.stack(level=0, dropna=False)

        if not get_max_profit:
            # Looks like the strikes, opens, and margins are all we need
            leg1_df = open_margins.to_frame()
        else:
            # Do some magic to get the maximum profits
            open_credits = bid_df.sub(leg1_asks, axis='rows')
            close_credits = (-bid_df).add(leg1_bids, axis='rows')

            for leg2_strike in close_credits.columns:
                # Get the forward-looking maximum by reversing the values,
                # applying a cumulative fmax (to ignore NaN), then flipping the
                # result
                #       1, NaN,   2,   1,   5,   0
                #       0,   5,   1,   2, NaN,   1   (flip)
                #       0,   5,   5,   5,   5,   5   (cummax)
                #       5,   5,   5,   5,   5,   0   (flip)
                reverse_closes = np.flip(
                    close_credits[leg2_strike].values)
                close_credits[leg2_strike] = np.flip(
                    np.fmax.accumulate(reverse_closes))

            max_profits = open_credits + close_credits
            max_profits = max_profits.stack(level=0, dropna=False)
            leg1_df = pd.concat((open_margins, max_profits), axis=1)

        # Strip out the trades that we can't actually open (i.e., one or both
        # sides weren't there or the combination was too pricey)
        leg1_df.dropna(inplace=True)

        # If there's nothing left, then we can just skip this leg 1 strike
        total_trades = leg1_df.shape[0]
        if verbose:
            print(
                '{:>2}: count({}) = {}'.format(
                    id, int(leg1_strike), total_trades)
            )
        if total_trades == 0:
            continue

        # Bring all of the indices into the dataframe to be columns
        leg1_df.reset_index(level=[0, 1], inplace=True)

        # The column names are all funky after the reset_index
        rename_dict = {
            'datetime': 'open_time',
            'strike': 'leg2_strike',
            0: 'open_margin',
        }
        if get_max_profit:
            rename_dict[1] = 'max_profit'
        leg1_df = leg1_df.rename(columns=rename_dict)

        # Add in the leg 1 strike
        leg1_strikes = np.full(total_trades, leg1_strike)
        leg1_df.insert(0, 'leg1_strike', leg1_strikes)

        all_open_times = leg1_df.open_time
        leg2_strikes   = leg1_df.leg2_strike

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
            columns = {k: 'leg1_' + k for k in leg1_meta.keys()}, inplace=True)
        leg2_meta.rename(
            columns = {k: 'leg2_' + k for k in leg2_meta.keys()}, inplace=True)

        result_dataframes.append(
            pd.concat((leg1_df.copy(), leg1_meta, leg2_meta), axis=1))

    profits_q.put(pd.concat(result_dataframes))

    if verbose:
        print('{:>2}: done'.format(id))

def collect_spreads(
    ticker,
    expiry,
    options_df=None,
    vertical=True,
    num_procs=10,
    max_margin=config.MARGIN,
    get_max_profit=False,
    verbose=False,
    debug=False
):
    # Build the thread-specific objects. We may not be using threads, but even
    # in this case the same thread objects can be used.

    # First we need a Queue of the strikes to be used for the buying leg.
    buy_strikes_q = multiprocessing.Queue()

    # And the threads will put their resulting DataFrames into this queue
    profits_q = multiprocessing.Queue()

    if options_df is None:
        options_df = utils.load_options(ticker, expiry)

    result_df_list = []

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

        spread_list = []
        if not debug:
            processes = []
            for i in range(num_procs):
                p = multiprocessing.Process(
                    target=spread_worker,
                    args=(
                        i,
                        option_type_df,
                        bid_df,
                        ask_df,
                        buy_strikes_q,
                        profits_q,
                        get_max_profit,
                        max_margin,
                        verbose,
                    )
                )
                p.start()
                processes.append(p)

            # This is going to be a lot of buffered data in the Queue. The
            # poster on this topic https://stackoverflow.com/a/26738946
            # mentioned that we may need to read from the Queue *before* joining
            # Maybe the better solution is to reference a semaphor
            for _ in range(num_procs):
                spread_list.append(profits_q.get())

            for p in processes:
                p.join()

            spread_df = pd.concat(spread_list)
        else:
            spread_worker(
                0,
                option_type_df,
                bid_df,
                ask_df,
                buy_strikes_q,
                profits_q,
                get_max_profit,
                max_margin,
                verbose,
            )
            spread_df = profits_q.get()

        # Add in a column showing which option type was in use for each leg.
        # Use the values of 1 and -1 to that 0 can be used to signify an empty
        # leg when working with the model
        type_array = np.ones(spread_df.shape[0])
        for leg in ['leg1', 'leg2']:
            spread_df.insert(
                0,
                leg + '_type',
                (-1 if o == 'P' else 1) * type_array
            )

        result_df_list.append(spread_df)

    result_df = pd.concat(result_df_list)

    if verbose:
        print('Adding underlying price')

    qs = QuestradeSecurities()
    candles = qs.get_candlesticks(
        ticker,
        result_df.open_time.min()-timedelta(minutes=5),
        result_df.open_time.max(),
        'FiveMinutes'
    )
    prices_df = pd.DataFrame(
        data = [c['open'] for c in candles],
        index = pd.to_datetime([c['end'] for c in candles]),
    )

    result_df['stock_price'] = prices_df.loc[result_df.open_time].values[:,0]

    # Here, too, we need to be aware of timezones, since there may have been
    # a DST <-> EST shift somewhere in the data
    if verbose:
        print('Converting open time to minutes to expiry')

    # Convert open_time to minutes_to_expiry.
    expiry_dt = datetime.strptime(expiry, '%Y-%m-%d') + timedelta(hours=16)

    # Add the time zone (is the next change in the fall or spring?)
    transitions = pytz.timezone('America/Toronto')._utc_transition_times
    dst = next(t for t in transitions if t > expiry_dt).month > 9
    expiry_dt = expiry_dt.replace(
        tzinfo = timezone(timedelta(hours = -4 if dst else -5)))


    # Get the epxiry as a UTC timedelta
    epoch = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    epoch_expiry = expiry_dt - epoch

    # Get the opens as a UTC timedelta
    epoch_opens = result_df.open_time.apply(
        lambda x: x.astimezone(timezone.utc) - epoch)

    result_df.drop('open_time', axis=1, inplace=True)
    result_df.insert(
        0,
        'minutes_to_expiry',
        (epoch_expiry - epoch_opens).apply(lambda x: x.total_seconds()) // 60
    )

    # Show the true values of the trade
    to_centify = ['open_margin']
    if get_max_profit:
        to_centify.append('max_profit')
    for k in to_centify:
        result_df[k] *= 100

    return result_df.reset_index(drop=True)

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

