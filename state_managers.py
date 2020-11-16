import numpy as np
import pandas as pd

import config
from   spread_processing import reinforcement_learning as rl
import utils

HEADER_COLS = [
    'optionType',
    'expiryDoW',
    'expiryWoM',
    'minsToExpiry',
    'stockPrice',
]
LEGS_COLS = [
    'long1_strike',
    'long1_askPrice',
    'long1_bidPrice',
    'long1_volume',
    'long1_volatility',
    'long1_openInterest',
    'long1_delta',
    'long1_gamma',
    'long1_theta',
    'long1_vega',
    'long1_rho',
    'long2_strike',
    'long2_askPrice',
    'long2_bidPrice',
    'long2_volume',
    'long2_volatility',
    'long2_openInterest',
    'long2_delta',
    'long2_gamma',
    'long2_theta',
    'long2_vega',
    'long2_rho',
    'short1_strike',
    'short1_askPrice',
    'short1_bidPrice',
    'short1_volume',
    'short1_volatility',
    'short1_openInterest',
    'short1_delta',
    'short1_gamma',
    'short1_theta',
    'short1_vega',
    'short1_rho',
    'short2_strike',
    'short2_askPrice',
    'short2_bidPrice',
    'short2_volume',
    'short2_volatility',
    'short2_openInterest',
    'short2_delta',
    'short2_gamma',
    'short2_theta',
    'short2_vega',
    'short2_rho',
]
OPEN_MULT = [-1, -1, 1, 1]
OPEN_COLS = [
    'long1_askPrice',
    'long2_askPrice',
    'short1_bidPrice',
    'short2_bidPrice'
]
CLOSE_MULT = [1, 1, -1, -1]
CLOSE_COLS = [
    'long1_bidPrice',
    'long2_bidPrice',
    'short1_askPrice',
    'short2_askPrice'
]

class StrategyStateManager(object):
    def __init__(
        self,
        ticker,
        expiries,
        max_margin=config.MARGIN,
        vertical=True,
        butterfly=True
    ):
        # Load in the options for this ticker-expiry
        self._options_df = pd.concat(
            (utils.load_options(ticker, exp) for exp in expiries))

        # We want to fill in the blanks when determining profits.
        ask_prices = self._options_df.askPrice
        bid_prices = self._options_df.bidPrice
        self._mean_ratio = 1 - ((ask_prices - bid_prices) / ask_prices).mean()

        # The maximum acceptable price for a trade that we will allow the agent
        # to make.
        self._max_margin = max_margin

        # The generator which is used to load up the next strategy
        self._strats_df_generator = rl.spreads_generator(
            self._options_df, vertical=vertical, butterfly=butterfly)

        # Collect the statistics for each row
        standardize_df = self._options_df[[
            'volume',
            'volatility',
            'delta',
            'gamma',
            'theta',
            'vega',
            'rho',
            'openInterest'
        ]]
        self._stds = standardize_df.var().pow(1/2)
        self._means = standardize_df.mean()

        # The DataFrame holding the timepoints for the current strategy.
        self._strat_df = None
        self._total_rows = None
        self._current_df_row = None
        self._current_df_index = None

        # We log-normalize the price values, but we still want to keep track
        # of when there are NaN values, implying that a specific leg cannot
        # be used to make a trade.
        self._nan_bid_ask_df = None

        # The maximum-attainable profit for this stratregy. The environment
        # will use this for determining the reward to provide to the agent.
        self.max_profit = None

        # The number of legs in use in the current strategy.
        self._leg_count = None

        # The column names that we will use to open and close trades.
        self._open_cols = None
        self._close_cols = None

        # The multipliers to apply to the above columns.
        self._open_mult = None
        self._close_mult = None

        # A numpy representation of the current timepoint plus
        # the metadata of the strategy when it was purchased.
        self.state = None

        # Indices for the state.
        # Is the trade available?
        self._available_i = 0

        # Is the trade affordable?
        self._good_margin_i = self._available_i + 1

        # Are we holding a trade?
        self._holding_i = self._good_margin_i + 1

        # Is this a four-legged strategy? (As opposed to two-legged).
        self._four_legs_i = self._holding_i + 1

        # The start of the header columns.
        self._header_i = self._four_legs_i + 1

        # The start of the current legs metadata columns.
        self._legs_i = self._header_i + len(HEADER_COLS)

        # The start of the held trade columns.
        self._held_i = self._legs_i + len(LEGS_COLS)

        # The [ask_price, ask_price, bid_price, bid_price] for the
        # strategy when it was purchased.
        self._held_open_prices = None

        # Reset to load everything up.
        self.reset()

    def _clean_df(self):
        # Make sure we're not considering times after the expiry.
        valid_times = self._strat_df.index <= self._strat_df.expiry[0]
        self._strat_df = self._strat_df[valid_times]

    def _process_df(self):
        all_columns = self._strat_df.columns
        log_cols = []

        # Pop out the expiry so that we don't need to care about it as we
        # process the other columns.
        expiry = self._strat_df.pop('expiry')[0]

        # Move on to taking the log(1+x) of each of the dollar-related or
        # quantity- related columns. Prior to taking the logs, we need to make
        # sure that all strikes and prices are on the same scale.
        for col_substring in ('askPrice', 'bidPrice'):
            log_cols += [col for col in all_columns if col_substring in col]
        self._strat_df.loc[:, log_cols] *= 100

        # Insert a column which is the open_time in the form of an integer
        # representing the number of minutes until expiry
        def get_minutes_to_expiry(x):
            return (utils.get_epoch_timestamp(expiry)
                        - utils.get_epoch_timestamp(x.name)) / 60
        self._strat_df.insert(
            0,
            'minsToExpiry',
            self._strat_df.apply(get_minutes_to_expiry, axis=1)
        )

        # Add the remaining log columns to the list.
        log_cols += ['stockPrice', 'minsToExpiry']
        for col_substring in ('strike', 'volume', 'Interest'):
            log_cols += [col for col in all_columns if col_substring in col]

        self._strat_df.loc[:, log_cols] = np.log1p(
            self._strat_df[log_cols].fillna(0))

        # Convert [call, put] column to 0,1 respectively.
        is_call = self._strat_df.optionType == 'C'
        self._strat_df.optionType = is_call.astype(np.int16)

        # For the remainder that are non-zero, standardize.
        std_cols = set(all_columns) - set(log_cols + ['optionType', 'expiry'])
        std_cols = [c for c in std_cols if self._strat_df[c].abs().max()]
        for greek in ['volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']:
            greek_cols = [c for c in std_cols if greek in c]
            zero_mean = self._strat_df[greek_cols] - self._means[greek]
            self._strat_df.loc[:,greek_cols] = zero_mean / self._stds[greek]

        # Now that we're finished normalizing, we can add in the processed
        # expiry data.
        # min-max expiry day of week (min = 1, max = 5)
        self._strat_df.insert(
            0,
            'expiryDoW',
            (expiry.isoweekday() - 1) / 4
        )
        # min-max expiry week of month (min = 1, max = 5)
        self._strat_df.insert(
            0,
            'expiryWoM',
            (
                int(
                    np.floor(
                        (expiry.day - expiry.weekday() + 3.9) / 7
                    ) + 1
                ) - 1
            ) / 4
        )

        # Put the columns in a standard order so that the agent can learn.
        self._strat_df = self._strat_df[HEADER_COLS + LEGS_COLS]

    def _set_legs_meta(self):
        selector = self._strat_df[OPEN_COLS].max() > 0
        self._leg_count = sum(selector)
        self._open_cols = np.array(OPEN_COLS)[selector]
        self._open_mult = np.array(OPEN_MULT)[selector]
        self._close_cols = np.array(CLOSE_COLS)[selector]
        self._close_mult = np.array(CLOSE_MULT)[selector]

    def _get_fees(self):
        return -2 * (config.BASE_FEE + self._leg_count / 100)

    def _set_max_profit(self):
        # NOTE: only consider columns actually used for the trade (e.g., ignore
        #       some for vertical).
        opens = self._strat_df.loc[:,self._open_cols]
        closes = self._strat_df.loc[:,self._close_cols]

        open_credits = (opens * self._open_mult).sum(axis=1, skipna=False)
        close_credits = (closes * self._close_mult).sum(axis=1, skipna=False)

        # Ignore the trades that we don't allow the agent to make because
        # they're too expensive.
        too_expensive = opens.sum(axis=1, skipna=False) > self._max_margin
        open_credits[too_expensive] = np.nan

        # Find the maximum credits for closing at or after each time step.
        close_credits = np.flip(np.fmax.accumulate(np.flip(close_credits)))

        credit_sum = open_credits + close_credits
        self.max_profit = credit_sum.max() - self._get_fees()

        if np.isnan(self.max_profit):
            # The only available closes were before the first available open.
            # Call it a wash.
            self.max_profit = 0

    def reset(self):
        # Move to the next strategy.
        self._strat_df = next(self._strats_df_generator)

        self._clean_df()

        # Flesh out the details of how we will make opening and closing trades.
        self._set_legs_meta()

        # Figure out the maximum profit that the Agent could obtain for this
        # strategy.
        self._set_max_profit()

        trade_cols = np.concatenate((self._open_cols, self._close_cols))

        # Fill in the NaN blanks with a reasonable approximation at bids and
        # asks.
        for col in ('short', 'long'):
            for num in [1,2]:
                bid_col = '{}{}_bidPrice'.format(col, num)
                ask_col = '{}{}_askPrice'.format(col, num)

                if bid_col not in trade_cols:
                    continue

                bids = self._strat_df[bid_col]
                asks = self._strat_df[ask_col]

                nan_bids = bids.isna()
                nan_asks = asks.isna()

                fill_bids = nan_bids & ~nan_asks
                fill_asks = nan_asks & ~nan_bids

                if fill_bids.any():
                    fill_rows = self._strat_df.loc[fill_bids]
                    fill_values = fill_rows[ask_col] * self._mean_ratio
                    self._strat_df.loc[fill_bids, bid_col] = fill_values
                if fill_asks.any():
                    fill_rows = self._strat_df.loc[fill_asks]
                    fill_values = fill_rows / self._mean_ratio
                    self._strat_df.loc[fill_asks, ask_col] = fill_values

        # Keep a copy of the original bid and ask prices to determine whether a
        # trade is available at particular timepoints.
        self._nan_bid_ask_df = self._strat_df.loc[:, trade_cols]

        # Normalize the DataFrame to be nice to the Agent.
        self._process_df()

        # The agent starts off not holding a strategy.
        self._held_open_prices = None

        # If this is our first run, initialize the numpy array representing
        # the current state and gather the index of the start of the held
        # strategy
        if self.state is None:
            # Make a state array that has just enough space for the expiry
            # metadata, current state and held strategy.
            self.state = np.zeros(self._held_i + len(LEGS_COLS))

        # Set the bit to let the agent know if this is a two- or four-
        # legged trade.
        self.state[self._four_legs_i] = self._leg_count == 4

        # Get the first datapoint for this strategy.
        self._current_df_index = None
        return self.step()

    def step(self):
        # Shift to the next datepoint for the strategy.
        try:
            self._current_df_index += 1
        except TypeError:
            self._current_df_index = 0
        self._current_df_row = self._strat_df.iloc[self._current_df_index]

        # Update the numpy array representing the current state.
        self.state[self._header_i:self._held_i] = (
            self._current_df_row.fillna(0).to_numpy()
        )

        # We can't actually buy/sell this trade if any of the legs are not
        # being sold or requested. That is, leg 1 and leg 2 must have non-NaN
        # `askPrice`s and leg 3 and leg 4 must have non-NoN `bidPrice`s.
        nan_prices = self._nan_bid_ask_df.iloc[self._current_df_index]
        if self.holding:
            # The legs that would be used to sell the trade.
            to_trade = nan_prices[self._close_cols]
            # There are no margin requirements for selling.
            self.state[self._good_margin_i] = 1
        else:
            # The legs that would be used to buy the trade.
            to_trade = nan_prices[self._open_cols]
            # There are margin requirements for buying though.
            self.state[self._good_margin_i] = (
                to_trade.sum(skipna=False) <= self._max_margin
            )

        # We only consider existing offers.
        self.state[self._available_i] = not to_trade.isna().any()

        return self.state

    def buy(self):
        if self.holding:
            raise ValueError('Already holding a trade.')

        if not self.state[self._available_i]:
            raise AttributeError('The full trade is not available')
        if not self.state[self._good_margin_i]:
            raise ValueError('The trade is too expensive.')

        # Flip the bit to let the agent know that it's holding a strategy.
        self.state[self._holding_i] = 1

        # Save the specific elements from the row that we will need to
        # reference when we finally sell the strategy.
        self._held_open_prices = self._current_df_row[self._open_cols]

        # Take the current strategy data and save it at the end of the state
        # so that the agent can reference it.
        self.state[self._held_i:] = self.state[self._legs_i: self._held_i]

    def sell(self, allow_nan=False):
        '''Return the profit made from buying an selling the strategy.'''
        if not self.holding:
            raise Exception('Not currently holding a strategy.')

        if not self.state[self._available_i]:
            if allow_nan:
                # Assume worthlessness of non-existent trades.
                # NOTE: may need to rethink this because maybe it's just way
                #       too expensive and that's why no one ever offered it.
                to_close = self._current_df_row.fillna(0)
            else:
                raise AttributeError('The full closing trade is not available')
        else:
            to_close = self._current_df_row

        # Start calculating the profit by determining how much we pay the
        # broker to open and close the trade.
        profit = self._get_fees()
        profit += (self._held_open_prices[self._open_cols]
                    * self._open_mult).sum()
        profit += (to_close[self._close_cols] * self._close_mult).sum()

        return profit

    @property
    def over(self):
        return self._current_df_index == self._strat_df.shape[0] - 1

    @property
    def holding(self):
        return self._held_open_prices is not None

