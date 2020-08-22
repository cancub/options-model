import datetime as dt
import numpy as np
import os
import pandas as pd
import subprocess as sp

import trade_processing as tp
import utils
import config

TICKER='SPY'
EXPIRY='2020-08-21'
MAX_MARGIN = 500

# Get the trades as well as the full metadata for each of the options
all_trades_df = utils.load_spreads(TICKER, EXPIRY)
options = utils.retrieve_options(TICKER, EXPIRY)

# Get the trades that are within out budget
viable_trades = all_trades_df[all_trades_df.open_margin < MAX_MARGIN]
total_trades = len(viable_trades)

# Determine the max profits when purchasing one of these trades
max_profit_with_fees = (viable_trades['max_profit']
                            - utils.calculate_fee(1, both_sides=True))

losers = viable_trades[max_profit_with_fees <= 0]
winners = viable_trades[max_profit_with_fees > 0]
total_winners = len(winners)
total_losers = len(losers)

print(
    '{} ({:.1%}) winners\n{} ({:.1%}) losers'.format(
        total_winners, total_winners / total_trades,
        total_losers, total_losers / total_trades
    )
)

# Get at most a 3 to 1 ratio of losers to winners, using the losers that were
# closest to profit
losers_to_get = total_winners * 3
hard_losers = losers.sort_values(
    by='max_profit', ascending=False)[:losers_to_get]

# Store whether the trade won or lost
hard_losers['winner'] = False
winners['winner'] = True
df = pd.concat((hard_losers, winners))

# Leaving around the side of the trade we're not working with not only doesn't
# make sense, but can leave around NaNs
options_l1 = options.copy().drop('bidPrice', 1)
options_l2 = options.copy().drop('askPrice', 1)

# We're paying for it, after all
options_l1.askPrice *= -1

examples = []
labels = []

print('Building data')
for index, trade in df.iterrows():
    # Now build up the X (strike, open credit, all metadata)
    examples.append(
        options_l1.loc[(index, trade.type, abs(trade.leg1_strike))].tolist()
            + options_l2.loc[(index, trade.type, trade.leg2_strike)].tolist()
    )
    labels.append(trade.winner)

print('Making array')
examples = np.array(examples)
labels = np.array(labels)

print('Normalizing')
mean = examples.mean(axis=0)
std = np.std(examples, axis=0)
examples = (examples - mean) / std

print('Concatenating X and Y')
data = np.concatenate((examples, np.atleast_2d(labels).T), axis=1)

print('Shuffling')
np.random.shuffle(data)
n_train = int(0.7 *  data.shape[0])

print('Saving')
if not os.path.exists(config.DATA_DIR):
    os.makedirs(config.DATA_DIR)
np.savez(
    os.path.join(config.DATA_DIR, '{}_{}.npz'.format(TICKER, EXPIRY)),
    train_examples=data[:n_train, :-1], train_labels=data[:n_train, -1],
    test_examples=data[n_train:, :-1], test_labels=data[n_train:, -1],
)
