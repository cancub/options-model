#! /usr/bin/env python
import tensorflow as tf

import argparse
from   io import BytesIO
import json
import numpy as np
import os
import pandas as pd
import random
import subprocess
import tempfile
import uuid

import config
import op_stats
import utils

class OptionsDataset(object):

    _files = ['train.bz2', 'val.bz2', 'test.bz2', 'metadata']

    def __init__(self, ticker, train_df, val_df, test_df, metadata, id=None):
        self.metadata = metadata
        self._id       = id or uuid.uuid4()
        self._train_df = train_df
        self._val_df   = val_df
        self._test_df  = test_df
        self._dir      = os.path.join(config.ML_DATA_DIR, ticker)

    def get_dataset_path(self):
        return os.path.join(self._dir, '{}.tar'.format(self._id))

    def _extract_file_to_bytes_fd(self, filename):
        return BytesIO(subprocess.check_output(
            ['tar', '-xOf', self.get_dataset_path(), filename]))

    @property
    def n_train(self):
        return self._train_df.shape[0]
    @property
    def n_val(self):
        return self._val_df.shape[0]
    @property
    def n_test(self):
        return self._test_df.shape[0]

    @property
    def n_features(self):
        return self._train_df.shape[1]

    def _normalize(
        self,
        df,
        means=None,
        stds=None,
        feature_order=None,
        min_max_cols=config.MIN_MAX_NORM_COLS,
        log_cols=config.LOG_NORM_COLS
    ):
        # Fill in the blanks.
        if None in (means, stds):
            loaded_means, loaded_vars = op_stats.pool_stats_from_stats_df(
                self.metadata['ticker']
            )

        means = means or loaded_means
        stds = stds or loaded_vars.pow(1/2)

        feature_order = feature_order or means.index.tolist()

        # First re-order the columns to fit the mode and perform the
        # normalization step
        df = (df[feature_order] - means[feature_order]) / stds[feature_order]

        # Then do min-max scaling on the columns that require it.
        if len(min_max_cols) > 0:
            to_mm_norm = [c for c in min_max_cols if c in df.columns]
            mm_df = df[to_mm_norm]
            c_mins = mm_df.min()
            df[to_mm_norm] = (mm_df - c_mins) / (mm_df.max() - c_mins)

        # Finally, do log scaling on all of the columns that require it.
        if len(log_cols) > 0:
            to_log_norm = [c for c in log_cols if c in df.columns]
            df[to_log_norm] = np.log(df[to_log_norm])

        return df

    def _build_dataset(self, df, *args, tf_dataset=True, **kwargs):
        X_norm = self._normalize(df, *args, **kwargs)
        Y = df.max_profit >= self.metadata['min_profit']
        if tf_dataset:
            return tf.data.Dataset.from_tensor_slices(
                (X_norm.values, Y.values))
        else:
            return X_norm, Y

    def get_train_set(self, *args, **kwargs):
        return self._build_dataset(self._train_df, *args, **kwargs)

    def get_val_set(self, *args, **kwargs):
        return self._build_dataset(self._val_df, *args, **kwargs)

    def get_test_set(self, *args, **kwargs):
        return self._build_dataset(self._test_df, *args, **kwargs)

    def get_sets(self, *args, **kwargs):
        return (
            self.get_train_set(*args, **kwargs),
            self.get_val_set(*args, **kwargs),
            self.get_test_set(*args, **kwargs),
        )

    @classmethod
    def from_tarball(cls, ticker, id):

        dataset_fpath = os.path.join(
            config.ML_DATA_DIR, ticker, '{}.tar'.format(id))

        with tempfile.TemporaryDirectory(prefix='options_data-') as tmpdir:

            tarball_files = utils.extract_and_get_file_list(
                dataset_fpath, tmpdir)

            # Make sure all the files we need are there
            for file in cls._files:
                assert(file in tarball_files)

            # Load the various datasets
            train = pd.read_pickle(os.path.join(tmpdir, 'train.bz2'))
            val = pd.read_pickle(os.path.join(tmpdir, 'val.bz2'))
            test = pd.read_pickle(os.path.join(tmpdir, 'test.bz2'))

            # ALso keep the metadata on hand
            with open(os.path.join(tmpdir, 'metadata'), 'r') as MF:
                metadata = json.load(MF)

        return cls(ticker, train, val, test, metadata, id)

    def to_tarball(self):
        with tempfile.TemporaryDirectory(prefix='options_data-') as tmpdir:
            # Add the data to the archive
            self._train_df.to_pickle(
                os.path.join(tmpdir, 'train.bz2'),
                protocol=config.PICKLE_PROTOCOL
            )
            self._val_df.to_pickle(
                os.path.join(tmpdir, 'val.bz2'),
                protocol=config.PICKLE_PROTOCOL
            )
            self._test_df.to_pickle(
                os.path.join(tmpdir, 'test.bz2'),
                protocol=config.PICKLE_PROTOCOL
            )

            # Add the metadata to the archive
            with open(os.path.join(tmpdir, 'metadata'), 'w') as MF:
                json.dump(self.metadata, MF)

            # Build the tarball
            out_path = self.get_dataset_path()
            subprocess.check_call(
                ['tar', '-C', tmpdir, '-cf', out_path] + self._files)


class StrategyGenerator(object):
    def __init__(self, min_profit, max_margin=None, randomize_legs=False):
        self._max_margin = max_margin
        self._min_profit = min_profit
        self._randomize_legs = randomize_legs
        self._generators = {}
        self._current_gen = None
        self._current_key = None

    @property
    def names(self):
        return list(self._generators.keys())

    @property
    def total_strategies(self):
        return len(self._generators)

    def _strat_paths_to_generator(self, strat_paths):
        np.random.shuffle(strat_paths)
        for p in strat_paths:
            yield pd.read_pickle(p)

    def add_generators(self, strats_paths):

        def path_to_name(p):
            return os.path.split(p)[1].split('-')[0]

        strats_dicts = {}

        # Walk through the files and append them to their respective list of
        # strategy DataFrame paths in the dictionary
        for p in strats_paths:
            try:
                strats_dicts[path_to_name(p)].append(p)
            except KeyError:
                strats_dicts[path_to_name(p)] = [p]

        for strat, paths in strats_dicts.items():
            if strat not in self.names:
                self._generators[strat] = []
            self._generators[strat].append(
                self._strat_paths_to_generator(paths))

    def shuffle(self):
        '''
        Randomize each list of generators.
        '''
        for name in self.names:
            random.shuffle(self._generators[name])

    def reset(self, key=None):
        '''
        Move on to a new generator and throw away the previous one.
        '''
        if key is None:
            if self._current_key is None:
                raise ValueError(
                    ('A key must be provided to begin using a '
                     'StrategyGenerator')
                )
            key = self._current_key
        else:
            self._current_key = key

        self._current_gen = self._generators[key].pop(0)

    def collect_data(self):

        # Keep searching until we get a dataframe with applicable data
        while True:
            try:
                df = next(self._current_gen)
            except StopIteration:
                # We're exhausted the current generator, so move on to the
                # next one
                self.reset()
                df = next(self._current_gen)

            if self._max_margin is not None:
                df = df[df.open_margin <= self._max_margin]

            if df.shape[0] != 0:
                # Ok, we've got something to return
                break

        winning_indices = df.max_profit >= self._min_profit

        return df[winning_indices], df[~winning_indices]


def build_dataset(
    ticker,
    max_margin=None,
    min_profit=0,
    total_datapoints=1*10**6,
    train_val_test_split=[0.9, 0.05, 0.05],
    loss_ratio=1,
    randomize_legs=False,
    hard_winners=False,
    win_pool_multiplier=1,
    hard_losers=True,
    loss_pool_multiplier=1,
    save=False,
    verbose=False,
):

    # We want to store the metadata about the data along with the data, which
    # means we must collect the locals now.
    metadata = locals()

    # Get rid of some of the unnecessary items. Deletion is the more robust way
    # to go about building the final metadata dict, since more metadata will
    # likely be added or removed later on, but the below information is likely
    # to be static.
    for k in ['save', 'verbose', 'train_val_test_split']:
        del metadata[k]

    def log(msg):
        if verbose:
            print(msg)

    def get_dataframe(strat_gen, n_examples):

        # We must add the fee for each trade (open and close) to
        strat_names = strat_gen.names
        win_dict = {n: None for n in strat_names}
        loss_dict = {n: None for n in strat_names}
        enough_wins = False
        enough_losses = False

        win_frac = 1/(loss_ratio + 1)
        total_strategies = strat_gen.total_strategies

        # The count may be updated if there are not enough datapoints to
        # fulfill the original request.
        count = n_examples
        def get_vals():
            min_wins = int((count * win_frac) / total_strategies)
            min_losses = int((count * (1 - win_frac)) / total_strategies)
            win_pool_size = min_wins * \
                (1 if not hard_winners else win_pool_multiplier)
            loss_pool_size = min_losses * \
                (1 if not hard_losers else loss_pool_multiplier)
            return min_wins, min_losses, win_pool_size, loss_pool_size

        min_wins, min_losses, win_pool_size, loss_pool_size = get_vals()

        log('Min wins: {}'.format(min_wins))
        log('Min losses: {}'.format(min_losses))

        for strat in strat_names:
            log('Collecting {}'.format(strat.replace('_', ' ')))

            # Reset the global generator so that we know the next DataFrame
            # generated will contain the strategy we're looking for
            strat_gen.reset(strat)

            total_wins = 0
            total_losses = 0

            while True:

                # Only continue looking for trades of this type if we're missing
                # data, otherwise move on to the next type.
                enough_wins = total_wins >= win_pool_size
                enough_losses = total_losses >= loss_pool_size
                if enough_wins and enough_losses:
                    break

                # Collect the data and add what we're missing to its respective
                # dataframe.
                strat_wins, strat_losses = strat_gen.collect_data()
                if not enough_wins and strat_wins is not None:
                    win_dict[strat] = pd.concat((
                        win_dict[strat], strat_wins))
                if not enough_losses and strat_losses is not None:
                    loss_dict[strat] = pd.concat((
                        loss_dict[strat], strat_losses))

                # Update the counts to figure out which types of trades must be
                # collected next round.
                try:
                    total_wins = win_dict[strat].shape[0]
                except AttributeError:
                    pass
                try:
                    total_losses = loss_dict[strat].shape[0]
                except AttributeError:
                    pass

                log('\twins: {:<8} ({:.1%})\tlosses: {:<8} ({:.1%})'.format(
                    total_wins, total_wins / win_pool_size,
                    total_losses, total_losses / loss_pool_size))

            # If desired, sort the winners such that they are in order of
            # smallest max_profit to largest max_profit.
            if hard_winners:
                win_dict[strat].sort_values(
                    by='max_profit', ascending=True, inplace=True)
            win_dict[strat] = win_dict[strat][:min_wins]

            # If desired, sort the losers such that they are in order of largest
            # max_profit to smallest max_profit.
            if hard_losers:
                loss_dict[strat].sort_values(
                    by='max_profit', ascending=False, inplace=True)
            loss_dict[strat] = loss_dict[strat][:min_losses]

        # Concat all of the DataFrames for all of the strategy types and wins
        # and losses.
        df = pd.concat(
            [d for d in win_dict.values()] +
            [d for d in loss_dict.values()]
        )

        # Give the final DataFrame the expected columns.
        log('Processing trades')
        df = utils.process_trades_df(df)

        return df

    # Get the paths to the available spreads tarballs
    spreads_dir = os.path.join(config.SPREADS_DIR, ticker)
    stats_df = pd.read_pickle(os.path.join(spreads_dir, 'stats'))
    expiries = stats_df.index.unique(level=0).values
    random.shuffle(expiries)

    strat_gen = StrategyGenerator(
        min_profit=min_profit,
        max_margin=max_margin,
        randomize_legs=randomize_legs
    )

    # Now we want to turn all of these expiries into a dict of lists of
    # generators. The dict is keyed by strategy type and then each entry of the
    # lists is a generator for all dataframes for that strategy type for a
    # specific expiry.
    with tempfile.TemporaryDirectory(prefix='options_examples') as tmpdir:
        for e in expiries:
            # Figure out where this tarball exists
            tarball_path = os.path.join(spreads_dir, '{}.tar'.format(e))

            # Make a sub-directory just for this expiry so the utility function
            # can parse it into generators
            exp_tmp_dir = os.path.join(tmpdir, e)
            os.mkdir(exp_tmp_dir)

            # Get the full list of files from this expiry
            file_list = list(map(
                lambda f: os.path.join(exp_tmp_dir, f),
                utils.extract_and_get_file_list(tarball_path, exp_tmp_dir)
            ))

            # Add the generators for each strategy for this expiry into the
            # dictionary
            strat_gen.add_generators(file_list)

        strat_gen.shuffle()

        n_train = int(total_datapoints * train_val_test_split[0])
        n_val = int(total_datapoints * train_val_test_split[1])
        n_test = int(total_datapoints * train_val_test_split[2])

        log('Winning profit : ${:7>.2f}'.format(min_profit * 100))
        log('Max margin     : ${:7>.2f}'.format(max_margin * 100))

        # We now have all the generators we need, so use them to fulfil the
        # requirements of each of the consituent datasets.
        # NOTE: the get_dataframe() function pops out generators such that
        #       one strategy type for one expiry will only ever appear in one
        #       dataset. This prevents cross-contamination.
        log('Collecting {} training datapoints.'.format(n_train))
        train_df = get_dataframe(strat_gen, n_train)
        log('Collecting {} validation datapoints.'.format(n_val))
        val_df = get_dataframe(strat_gen, n_val)
        log('Collecting {} test datapoints.'.format(n_test))
        test_df = get_dataframe(strat_gen, n_test)

    metadata['n_train'] = train_df.shape[0]
    metadata['n_val'] = val_df.shape[0]
    metadata['n_test'] = test_df.shape[0]

    log('Creating OptionsDataset')

    # Generate up a dataset object
    dataset = OptionsDataset(ticker, train_df, val_df, test_df, metadata)

    # Save the final dataset if requested.
    if save:
        log('Saving dataset')
        dataset.to_tarball()

    return dataset

def get_dataset_details(ticker):
    data_dir = os.path.join(config.ML_DATA_DIR, ticker)
    ids = []
    datasets_meta = {}

    for fname in (f for f in os.listdir(data_dir) if f.endswith('.tar')):
        fpath = os.path.join(data_dir, fname)

        id = fname.split('.tar')[0]
        ids.append(id)

        # Get the metadata
        meta = json.loads(
            subprocess.check_output(['tar', '-xOf', fpath, 'metadata']))

        for k, v in meta.items():
            try:
                datasets_meta[k].append(v)
            except KeyError:
                datasets_meta[k] = [v]

    datasets_meta['id'] = ids

    return pd.DataFrame(
        datasets_meta).sort_values(by='id').reset_index(drop=True)

def load_dataset(ticker, id=None, **kwargs):
    verbose = kwargs.pop('verbose', False)
    def log(msg):
        if not verbose: return
        print(msg)

    datasets = get_dataset_details(ticker)

    if id is not None:
        if id not in datasets.id.values:
            raise Exception(
                'No dataset exists for {} with id {}'.format(ticker, id))
        return OptionsDataset.from_tarball(ticker, id)

    if datasets.shape[0] == 0:
        log('No saved datasets. Building.')
        # Make sure to save the data, since nothing currently exists.
        kwargs['save'] = True
        return build_dataset(ticker, **kwargs)

    # Narrow down by the money details
    datasets = datasets[
        (datasets.max_margin == kwargs.pop('max_margin', np.inf))
        & (datasets.min_profit == kwargs.pop('min_profit', 0))
    ]

    # Narrow down by the remaining arguments
    for key, val in kwargs.items():
        # Find datasets that are reasonably close to this value
        column = datasets[key]
        datasets = datasets[(column >= val / 1.1) & (column <= val * 1.1)]

    # Do we have anything that fits the bill?
    if datasets.shape[0] > 0:
        # Load the largest dataset we can find.
        log('Loading largest dataset.')
        info = datasets.iloc[datasets.total_datapoints.argmax()]
        log(info)
        return OptionsDataset.from_tarball(ticker, id=info.id)
    else:
        # Generate a stock-standard set of data if nothing is available.
        log('No available matching datasets. Building.')
        # Make sure to save the data, since nothing currently exists that meet
        # this criteria
        kwargs['save'] = True
        return build_dataset(ticker, verbose=verbose, **kwargs)

def _list(**kwargs):
    datasets = get_dataset_details(kwargs['ticker'])
    print(datasets)

def _build(**kwargs):

    # Do a bit of reformatting of the difficult and pooling args as well as the
    # data split
    kwargs['hard_winners'], kwargs['hard_losers'] = kwargs.pop('difficult')
    (kwargs['win_pool_multiplier'],
     kwargs['loss_pool_multiplier']) = kwargs.pop('pool_multiplier')
    kwargs['train_val_test_split'] = kwargs.pop('split')

    # Always save the dataset when being run from script.
    kwargs['save'] = True

    dataset = build_dataset(**kwargs)

    print(dataset.get_dataset_path())

def _get_parser():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_list = subparsers.add_parser('list')
    parser_list.add_argument('ticker', metavar='TICKER')
    parser_list.set_defaults(func=_list)

    parser_build = subparsers.add_parser('build')
    parser_build.add_argument(
        '-m', '--max-margin', type=float, default=5,
        help='The maximum margin to open the trade. (NOTE: per 100 trades)')
    parser_build.add_argument(
        '-w', '--winning-profit', type=float, default=1, dest='min_profit',
        help=('The dollar value threshold between what is considered a "win" '
              'and a "loss." NOTE: per 100 trades'))
    parser_build.add_argument(
        '-t', '--total-datapoints', type=int, default=1*10**6,
        help='The total number of examples that must be generated.')
    parser_build.add_argument(
        '-s', '--split', type=float, nargs=3, default=[0.9, 0.05, 0.05],
        help=('The percent values of the data to be reserved for train, '
              'validation and test, respectively. NOTE: these are decimal '
              'values, so use 0.9 rather than 90.'))
    parser_build.add_argument(
        '-l', '--loss-ratio', type=float, default=1,
        help='The number of losses to obtain for each win of a given strategy')
    parser_build.add_argument(
        '-r', '--randomize-legs', action='store_true',
        help=('Randomize the order in which the legs appear in the output '
              'data. NOTE: the order of the columns within leg sections remains'
              'unchanged.'))
    parser_build.add_argument(
        '-d', '--difficult', nargs=2, type=lambda x: bool(int(x)),
        default=[False, True],
        help=('When selecting the final group of trades for a strategy, only '
              'take the winners and/or losers that are the hardest to guess.'))
    parser_build.add_argument(
        '-p', '--pool-multiplier', type=int, nargs=2, default=[1, 1],
        help=('When gathering data for winners and losers, collect X times as '
              'must datapoints as needed, e.g., if 10K winners for a specific '
              'strategy are needed and the pool multiplier for winners is 3, '
              'collect 30K winners. NOTE: this is best used in conjunction '
              'with --hard, as it will collect more difficult-to-guess trades '
              'that would be obtained normally.'))
    parser_build.add_argument(
        '-v', '--verbose', action='store_true',
        help='Display the information about data collection.')
    parser_build.add_argument('ticker', metavar='TICKER')
    parser_build.set_defaults(func=_build)

    return parser

def main(args=None):
    parser = _get_parser()
    args = parser.parse_args(args)

    kwargs = vars(args)
    kwargs.pop('func')(**kwargs)

if __name__ == '__main__':
    main()
