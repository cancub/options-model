from collections import OrderedDict
import datetime as dt
import numpy as np
import os
import pandas as pd
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

def collect_TA(ticker, dates):
    # return the technical analysis portion for the ticker to be used in the
    # final input vector
    pass

# TODO:
# check fees on questrade for each
# check how to find vega

def _print_progress(exp, legs):
    leg_strs = []
    for i in range(len(legs)):
        leg = legs[i]
        leg_strs.append(
            'leg{}: {:<5} ({:>3}%)'.format(
                i, leg['strike'], int(leg['index']*100/leg['total']))
        )
    sys.stdout.write(
        '\r' + 'expiry: {}, '.format(exp) + ', '.join(leg_strs)
    )

def determine_profits(legs, margin_filter):
    n_legs = len(legs)
    def get_action_premiums(legs, action):
        '''
        leg: information about leg (position, df)
        action: 'open'/'close'
        '''
        concat_list = []
        for l in legs:
            # Figure out what's going on with this leg
            leg_df = l['bid_ask']
            position = l['position']

            # Detemine the corresponding column and credit calculation
            # multiplier
            if ((action == 'open' and position == 'long') or
                    (action == 'close' and position == 'short')):
                # We're looking to buy and will be debited
                col = 'ask'
                multiplier = -1
            elif ((action == 'open' and position == 'short') or
                    (action == 'close' and position == 'long')):
                # We're looking to sell and will be credited
                col = 'bid'
                multiplier = 1
            else:
                raise ValueError(
                    'Unknown combination of position ({}) and action ({})'.format(
                        position, action)
                )

            concat_list.append(leg_df[col] * multiplier)

        # Build the DataFrame and set the column names to the legs
        action_df = pd.concat(concat_list, axis=1)
        action_df.columns = ['leg{}'.format(i+1) for i in range(n_legs)]
        return action_df

    # Get all of the bid and ask values related to opening a position for each
    # of the legs.
    open_df = get_action_premiums(legs, 'open')
    # Get the last index so that we can remove it after we're done filtering
    last_time = open_df.index[-1]
    # Remove the rows of the open_df that correspond to trades that are outside
    # our margin requirements
    margin_filter(
        open_df, pd.concat([l['bid_ask']['ask'] for l in legs], axis=1))
    # Get rid of rows in which EITHER column is NaN, since we can't actually
    # open a position here
    open_df = open_df[np.isnan(open_df).sum(axis=1) == 0]
    # Finally, remove the row corresponding to the last trade that was available
    # in the original data (if it still exists). We can't actually use this
    # trade because there's no data available for how much it would cost to
    # close it out
    try:
        open_df.drop(last_time, inplace=True)
    except KeyError:
        # It was already removed by the filtering
        pass


    # Get all of the bid and ask values related to closing a position for each
    # of the legs. Note that we need to get rid of rows in which BOTH column
    # is NaN, since they're useless to us. We can still use rows where either
    # leg is non-NaN to find out the absolute maximum profit we could make if we
    # didn't need to close all legs sim
    close_df = get_action_premiums(legs, 'close')
    close_df = close_df[np.isnan(close_df).sum(axis=1) < n_legs]
    # For determining the max profit for a simultaneous close, we do actually
    # need to skip these partial rows
    sim_close_df = close_df[np.isnan(close_df).sum(axis=1) == 0]

    # Find the amount that we would be credited for opening the trade at each of
    # the datetimes and then ignore the ones that are out of our price range for
    # any of the legs
    element_viability = open_df > -MAX_DEBIT/100
    viable_open_times = open_df[element_viability.sum(axis=1) == n_legs].index

    if len(viable_open_times) == 0:
        return None

    # Just because we can open a trade, doesn't mean that we can close it.
    closable_open_times = []

    viable_leg_open_credits = open_df.loc[viable_open_times]
    leg_names = open_df.columns

    # Prepare the dictionary that will be used to build the DataFrame. Use an
    # ordered dict so that we maintain a nice order of columns
    df_data = OrderedDict()
    df_data['simul_exit_dt'] = []
    for name in leg_names:
        df_data['{}_entry_credit'.format(name)] = []
        df_data['{}_simul_exit_credit'.format(name)] = []
        df_data['{}_solo_exit_dt'.format(name)] = []
        df_data['{}_solo_exit_credit'.format(name)] = []

    for open_time in viable_open_times:
        # Get the DataFrames representing the closing credits for all datetimes
        # after the opening time
        all_closes = close_df.loc[open_time:].iloc[1:]
        sim_closes = sim_close_df.loc[open_time:].iloc[1:]

        # We can't close this trade if the closing side of one of the legs has
        # 0 offers
        if 0 in (np.isnan(all_closes) == False).sum().tolist():
            continue

        # This is for sure a closable trade
        closable_open_times.append(open_time)

        # Get the datetime of the absolute maximum for each leg
        individual_max_dts = all_closes.idxmax()

        # Get the datetime for the simultaneous maximum
        simultaneous_max_dt = sim_closes.sum(axis=1).idxmax()

        # add all the values for this row of the result dataframe
        df_data['simul_exit_dt'].append(simultaneous_max_dt)
        for name in leg_names:
            df_data['{}_entry_credit'.format(name)].append(
                viable_leg_open_credits.loc[open_time, name])
            df_data['{}_simul_exit_credit'.format(name)].append(
                sim_closes.loc[simultaneous_max_dt, name])
            df_data['{}_solo_exit_dt'.format(name)].append(
                individual_max_dts[name])
            df_data['{}_solo_exit_credit'.format(name)].append(
                all_closes.loc[individual_max_dts[name], name])

    try:
        return pd.DataFrame(df_data, index=closable_open_times)
    except:
        import pdb; pdb.set_trace()

def get_basic_datetimes(times_strings):
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
    stock_dtimes = get_basic_datetimes(all_stock_prices['datetime'])
    return all_stock_prices['ticker'].set_index(stock_dtimes, inplace=True)

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
        metadata = pd.read_pickle(base_path + '_meta')
        # In older versions of the metadata, indices do not correspond to the
        # option indices in the numpy array. Resetting them here fixes that and
        # does not affect newer metadata
        metadata.reset_index(inplace=True, drop=True)
        with open(base_path + '_times', 'r') as TF:
            dtimes = get_basic_datetimes(TF.read().split('\n')[:-1])

        # Make sure all the quantities jive
        assert(data.shape[0] == 10)
        assert(data.shape[1] == len(dtimes))
        assert(data.shape[2] == len(metadata))

        exp_date = dt.datetime.strptime(exp, '%Y-%m-%d').date()
        result[exp_date] = {
            'data': data, 'metadata': metadata, 'datetimes': dtimes}

    return result

# ============================== Vertical spreads ==============================

def bull_call_spreads(ticker, working_dir='pickles', vertical=True):
    '''
    A bull call spread is constructed by buying a call option with a lower
    strike price (K), and selling another call option with a higher strike
    price.

    Often the call with the lower exercise price will be at-the-money while the
    call with the higher exercise price is out-of-the-money. Both calls must
    have the same underlying security and expiration month. If the bull call
    spread is done so that both the sold and bought calls expire on the same
    day, it is a vertical debit call spread.

    Break even point= Lower strike price+ Net premium paid 
    '''
    def margin_filter(open_df, asks_df):
        # Get the market values of the spreads and use this to figure out if we
        # have enough margin for that trade
        open_df.drop(
            asks_df[asks_df.sum(axis=1)*100 > MARGIN].index, inplace=True)
        # TODO: what the hell is "the spread loss amount, if any, that would
        #       result if both options were exercised"        

    # Get all the necessary information to do some processing
    single_legs = collect_single_legs(ticker, working_dir)

    # TODO: use to whittle things down to only those calls call that are at or
    # above the money, as per the description
    # stock_prices = get_stock_prices(ticker, working_dir)

    concat_list = []

    for exp_date, exp_dict in single_legs.items():
        data = exp_dict['data']
        meta = exp_dict['metadata']
        dtimes = exp_dict['datetimes']

        # Focus exclusively on calls since this is a call spread
        call_df = meta[meta['type'] == 'C']
        for l1_index, leg1 in call_df.iterrows():
            # This is Leg 1, our lower strike call which we will be longing. Use
            # its metadata to grab the bid and ask prices from the data array.

            # Before we continue though, we need to make sure that we can even
            # afford this trade based on what we've said is the maximum we're
            # willing to spend for one leg
            l1_asks = data[1,:,l1_index]
            if np.nanmin(l1_asks) > MAX_DEBIT/100:
                continue

            l1_df = pd.DataFrame(
                {'bid': data[0,:,l1_index], 'ask': l1_asks},
                index=dtimes
            )
            # Walk through the remaining calls which have a higher strike, as
            # these will be our potential second legs
            all_leg2s_df = call_df[call_df['strike'] > leg1['strike']]

            for l2_index, leg2 in all_leg2s_df.iterrows():
                # This is Leg 2, our higher strike call which we will be
                # shorting. Use its metadata to grab the bid and ask prices from
                # the data array.
                l2_df = pd.DataFrame(
                    {'bid': data[0, :, l2_index], 'ask': data[1, :, l2_index]},
                    index=dtimes
                )

                # Finally, get the profits of all of the possible strategies
                # that could have been used for this combination of legs
                legs_profits = determine_profits(
                    [
                        {
                            'position': 'long',
                            'bid_ask': l1_df,
                            'strike': leg1['strike'],
                        },
                        {
                            'position': 'short',
                            'bid_ask': l2_df,
                            'strike': leg2['strike'],
                        },
                    ],
                    margin_filter
                )

                # A return value of None implies that there are no profitable
                # trades in our price range
                if legs_profits is None or len(legs_profits) == 0:
                    continue

                n_trades = len(legs_profits)

                # Add in the expiry columns and option indices for each leg
                legs_profits['leg1_expiry'] = np.array([exp_date] * n_trades)
                legs_profits['leg1_index'] = np.array([l1_index] * n_trades)
                legs_profits['leg2_expiry'] = np.array([exp_date] * n_trades)
                legs_profits['leg2_index'] = np.array([l2_index] * n_trades)

                concat_list.append(legs_profits)

    return pd.concat(concat_list)




def bull_put_spreads(single_legs, vertical = True):
    '''
    A bull put spread is constructed by selling higher striking in-the-money put
    options and buying the same number of lower striking out-of-the-money put
    options on the same underlying security with the same expiration date. The
    options trader employing this strategy hopes that the price of the
    underlying security goes up far enough that the written put options expire
    worthless.

    If the bull put spread is done so that both the sold and bought put expire
    on the same day, it is a vertical credit put spread.

    Break even point = upper strike price - net premium received 
    '''
    

    pass

def bear_call_spreads(ticker, vertical = True):
    '''
    A bear call spread is a limited profit, limited risk options trading
    strategy that can be used when the options trader is moderately bearish on
    the underlying security. It is entered by buying call options of a certain
    strike price and selling the same number of call options of lower strike
    price (in the money) on the same underlying security with the same
    expiration month. 
    '''
    pass

def bear_put_spreads(ticker, working_dir='pickles', vertical=True):
    '''
    A bear put spread is a limited profit, limited risk options trading strategy
    that can be used when the options trader is moderately bearish on the
    underlying security. It is entered by:

    - buying higher striking in-the-money put options and
    - selling the same number of lower striking out-of-the-money put options on
      the same underlying security and the same expiration month.
    '''

    def margin_filter(open_df, asks_df):
        # Get the market values of the spreads and use this to figure out if we
        # have enough margin for that trade
        open_df.drop(
            asks_df[asks_df.sum(axis=1)*100 > MARGIN].index, inplace=True)
        # TODO: what the hell is "the spread loss amount, if any, that would
        #       result if both options were exercised"        

    # Get all the necessary information to do some processing
    single_legs = collect_single_legs(ticker, working_dir)

    # TODO: use to whittle things down to only those puts put that are at or
    # above the money, as per the description
    # stock_prices = get_stock_prices(ticker, working_dir)

    concat_list = []

    for exp_date, exp_dict in single_legs.items():
        data = exp_dict['data']
        meta = exp_dict['metadata']
        dtimes = exp_dict['datetimes']

        # Focus exclusively on puts since this is a put spread
        put_df = meta[meta['type'] == 'P']
        leg1_count = len(put_df)
        for l1_index, leg1 in put_df.iterrows():
            # This is Leg 1, our lower strike put which we will be longing. Use
            # its metadata to grab the bid and ask prices from the data array.

            # Before we continue though, we need to make sure that we can even
            # afford this trade based on what we've said is the maximum we're
            # willing to spend for one leg
            l1_asks = data[1,:,l1_index]
            if np.nanmin(l1_asks) > MAX_DEBIT/100:
                continue

            l1_df = pd.DataFrame(
                {'bid': data[0,:,l1_index], 'ask': l1_asks},
                index=dtimes
            )
            # Walk through the remaining puts which have a lower strike, as
            # these will be our potential second legs
            all_leg2s_df = put_df[put_df['strike'] < leg1['strike']]
            for l2_index, leg2 in all_leg2s_df.iterrows():
                _print_progress(exp_date, [
                    {
                        'strike': leg1['strike'],
                        'index': l1_index,
                        'total': leg1_count
                    },
                    {
                        'strike': leg2['strike'],
                        'index': l2_index,
                        'total': all_leg2s_df.index[-1]
                    },
                ])
                # This is Leg 2, our higher strike put which we will be
                # shorting. Use its metadata to grab the bid and ask prices from
                # the data array.
                l2_df = pd.DataFrame(
                    {'bid': data[0, :, l2_index], 'ask': data[1, :, l2_index]},
                    index=dtimes
                )


                # Finally, get the profits of all of the possible strategies
                # that could have been used for this combination of legs
                legs_profits = determine_profits(
                    [
                        {
                            'position': 'long',
                            'bid_ask': l1_df,
                            'strike': leg1['strike'],
                        },
                        {
                            'position': 'short',
                            'bid_ask': l2_df,
                            'strike': leg2['strike'],
                        },
                    ],
                    margin_filter
                )

                # A return value of None implies that there are no profitable
                # trades in our price range
                if legs_profits is None or len(legs_profits) == 0:
                    continue

                n_trades = len(legs_profits)

                # Add in the expiry columns and option indices for each leg
                legs_profits['leg1_expiry'] = np.array([exp_date] * n_trades)
                legs_profits['leg1_index'] = np.array([l1_index] * n_trades)
                legs_profits['leg2_expiry'] = np.array([exp_date] * n_trades)
                legs_profits['leg2_index'] = np.array([l2_index] * n_trades)

                concat_list.append(legs_profits)

        return pd.concat(concat_list)

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

