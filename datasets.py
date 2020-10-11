#! /usr/bin/env python
import argparse
from   io import BytesIO
import json
import numpy as np
import os
import pandas as pd
import subprocess
import tempfile
import uuid

import config
import utils

class OptionsDataset(object):
    _files = ['dataframe', 'metadata']
    def __init__(self, ticker, df=None, id=None, metadata=None):
        self._data     = df
        self._metadata = metadata
        self._id       = id
        self._dir      = os.path.join(config.ML_DATA_DIR, ticker)
        self._labels   = None

    def get_dataset_path(self):
        return os.path.join(self._dir, '{}.tar'.format(self._id))

    def _extract_file_to_bytes_fd(self, filename):
        return BytesIO(subprocess.check_output(
            ['tar', '-xOf', self.get_dataset_path(), filename]))

    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_pickle(
                self._extract_file_to_bytes_fd('dataframe'),
                compression='bz2'
            )
        return self._data

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = json.load(
                self._extract_file_to_bytes_fd('metadata'))
        return self._metadata

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.data.max_profit >= self.metadata['min_profit']
        return self._labels

    def save(self, id=None):
        if id is None:
            if self._id is None:
                # Figure out what we should call ourselves in the filesystem
                self._id = uuid.uuid4()
        else:
            self._id = id

        with tempfile.TemporaryDirectory(prefix='options_data-') as tmpdir:
            # Add the data to the archive
            self._data.to_pickle(
                os.path.join(tmpdir, 'dataframe'),
                compression='bz2'
            )

            # Add the metadata to the archive
            with open(os.path.join(tmpdir, 'metadata'), 'w') as MF:
                json.dump(self._metadata, MF)

            # Build the tarball
            out_path = self.get_dataset_path()
            subprocess.check_call(
                ['tar', '-C', tmpdir, '-cf', out_path] + self._files)


def build_dataset(
    ticker,
    max_margin=None,
    min_profit=0,
    total_datapoints=1*10**6,
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
    for k in ['save', 'verbose']:
        del metadata[k]

    def log(msg):
        if verbose:
            print(msg)

    def strats_paths_to_generators(strats_paths):

        def strat_paths_to_generator(strat_paths):
            np.random.shuffle(strat_paths)
            for p in strat_paths:
                yield pd.read_pickle(p)

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

        return {k: strat_paths_to_generator(v) for k, v in strats_dicts.items()}

    # We must add the fee for each trade (open and close) to
    strategy_generators = {}
    strats_dfs = {'win': {}, 'loss': {}}
    enough_wins = False
    enough_losses = False

    log('Winning profit: {}'.format(min_profit * 100))

    # Get the paths to the available spreads tarballs
    data_dir = os.path.join(config.ML_DATA_DIR, ticker)
    tarball_paths = (os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if f.endswith('.tar'))

    # Extract all of the available strategy DataFrames for all available
    # expiries into one directory
    with tempfile.TemporaryDirectory(prefix='options_examples') as tmpdir:
        file_list = []
        for p in tarball_paths:
            # Extract it and get the file list
            file_list += list(map(
                lambda f: os.path.join(tmpdir, f),
                utils.extract_and_get_file_list(p, tmpdir)
            ))

        # Convert each list of strategies into a generator of DataFrames for
        # that strategy
        strategy_generators = strats_paths_to_generators(file_list)

        strat_names = list(strategy_generators.keys())

        # Now that we know the names of the unique strategies, we can count how
        # many there are ...
        total_strategies = len(strat_names)

        # ... determine how many winners and losers are needed for each
        # strategy ...
        win_frac = 1/(loss_ratio + 1)
        min_wins = int((total_datapoints * win_frac) / total_strategies)
        min_losses = int((total_datapoints * (1 - win_frac)) / total_strategies)

        win_pool_size = min_wins * \
            (1 if not hard_winners else win_pool_multiplier)
        loss_pool_size = min_losses * \
            (1 if not hard_losers else loss_pool_multiplier)

        log('Min wins: {}'.format(min_wins))
        log('Min losses: {}'.format(min_losses))

        # ... and prepare the win/loss dictionary with their names.
        for name in strat_names:
            strats_dfs['win'][name] = None
            strats_dfs['loss'][name] = None

        # A little helper function to get a new (win, lose) DataFrame tuple for
        # a specific strategy
        def collect_strategy_data(key):
            df = next(strategy_generators[key])

            if max_margin is not None:
                df = df[df.open_margin <= max_margin]

            if df.shape[0] == 0:
                return None, None

            winning_indices = df.max_profit >= min_profit

            # Randomize the leg order to make the model more robust
            if randomize_legs:
                utils.randomize_legs_columns(df)

            return df[winning_indices], df[~winning_indices]

        for strat in strat_names:
            log('Collecting {}'.format(strat.replace('_', ' ')))

            while True:

                # Figure out which types of trades must be collected
                try:
                    current_wins = strats_dfs['win'][strat].shape[0]
                except AttributeError:
                    current_wins = 0
                try:
                    current_losses = strats_dfs['loss'][strat].shape[0]
                except AttributeError:
                    current_losses = 0

                enough_wins = current_wins >= win_pool_size
                enough_losses = current_losses >= loss_pool_size

                if enough_wins and enough_losses:
                    break

                strat_wins, strat_losses = collect_strategy_data(strat)
                if not enough_wins and strat_wins is not None:
                    strats_dfs['win'][strat] = pd.concat((
                        strats_dfs['win'][strat], strat_wins))
                if not enough_losses and strat_losses is not None:
                    strats_dfs['loss'][strat] = pd.concat((
                        strats_dfs['loss'][strat], strat_losses))

                try:
                    total_wins = strats_dfs['win'][strat].shape[0]
                except AttributeError:
                    total_wins = 0
                try:
                    total_losses = strats_dfs['loss'][strat].shape[0]
                except AttributeError:
                    total_losses = 0
                log('\twins: {:<8} ({:.1%})\tlosses: {:<8} ({:.1%})'.format(
                    total_wins, total_wins / win_pool_size,
                    total_losses, total_losses / loss_pool_size))

            # If desired, sort the winners such that they are in order of smallest
            # max_profit to largest max_profit.
            if hard_winners:
                strats_dfs['win'][strat].sort_values(
                    by='max_profit', ascending=True, inplace=True)
            strats_dfs['win'][strat] = strats_dfs['win'][strat][:min_wins]
            # If desired, sort the losers such that they are in order of largest
            # max_profit to smallest max_profit.
            if hard_losers:
                strats_dfs['loss'][strat].sort_values(
                    by='max_profit', ascending=False, inplace=True)
            strats_dfs['loss'][strat] = strats_dfs['loss'][strat][:min_losses]

    # Concat all of the DataFrames
    df = pd.concat(
        [d for d in strats_dfs['win'].values()] +
        [d for d in strats_dfs['loss'].values()]
    )

    log('Processing trades')
    df = utils.process_trades_df(df)

    metadata['total_datapoints'] = df.shape[0]

    # Load up a dataset object
    dataset = OptionsDataset(ticker, df=df, metadata=metadata)

    # Save the final dataset if requested.
    if save:
        dataset.save()

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

def load_dataset(ticker, **kwargs):
    def log(msg):
        if not kwargs.get('verbose', False): return
        print(msg)

    datasets = get_dataset_details(ticker)

    if datasets.shape[0] == 0:
        log('No saved datasets. Building.')
        # Make sure to save the data, since nothing currently exists.
        kwargs['save'] = True
        return build_dataset(ticker, **kwargs)

    # Narrow down by the high-level details
    if 'max_margin' in kwargs:
        datasets = datasets[datasets.max_margin == kwargs['max_margin']]
    if 'min_profit' in kwargs:
        datasets = datasets[datasets.min_profit == kwargs['min_profit']]
    if 'total_datapoints' in kwargs:
        # Find a reasonably-close number of datapoints (within 10% on either
        # side)
        count = kwargs['total_datapoints']
        datasets = datasets[
            (datasets.total_datapoints >= count / 1.1)
                & (datasets.total_datapoints <= count * 1.1) ]
    if 'loss_ratio' in kwargs:
        datasets = datasets[datasets.loss_ratio == kwargs['loss_ratio']]

    # Do we have anything that fits the bill?
    if datasets.shape[0] > 0:
        log('Loading largest dataset.')
        # Load the largest dataset we can find.
        info = datasets.iloc[datasets.total_datapoints.argmax()]
        if kwargs.get('verbose', False):
            print(info)
        return OptionsDataset(ticker, id=info.id)
    else:
        # Generate a stock-standard set of data if nothing is available.
        log('No available matching datasets. Building.')
        # Make sure to save the data, since nothing currently exists that meet
        # this criteria
        kwargs['save'] = True
        return build_dataset(ticker, **kwargs)

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--max-margin', type=float, default=5,
        help='The maximum margin to open the trade. (NOTE: per 100 trades)')
    parser.add_argument(
        '-w', '--winning-profit', type=float, default=1, dest='min_profit',
        help=('The dollar value threshold between what is considered a "win" '
              'and a "loss." NOTE: per 100 trades'))
    parser.add_argument(
        '-t', '--total-datapoints', type=int, default=1*10**6,
        help='The total number of examples that must be generated.')
    parser.add_argument(
        '-l', '--loss-ratio', type=float, default=1,
        help='The number of losses to obtain for each win of a given strategy')
    parser.add_argument(
        '-r', '--randomize-legs', action='store_true',
        help=('Randomize the order in which the legs appear in the output '
              'data. NOTE: the order of the columns within leg sections remains'
              'unchanged.'))
    parser.add_argument(
        '-d', '--difficult', nargs=2, type=lambda x: bool(int(x)),
        default=[False, True],
        help=('When selecting the final group of trades for a strategy, only '
              'take the winners and/or losers that are the hardest to guess.'))
    parser.add_argument(
        '-p', '--pool-multiplier', type=int, nargs=2, default=[1, 1],
        help=('When gathering data for winners and losers, collect X times as '
              'must datapoints as needed, e.g., if 10K winners for a specific '
              'strategy are needed and the pool multiplier for winners is 3, '
              'collect 30K winners. NOTE: this is best used in conjunction '
              'with --hard, as it will collect more difficult-to-guess trades '
              'that would be obtained normally.'))
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Display the information about data collection.')
    parser.add_argument('ticker', metavar='TICKER')

    return parser

def main(args=None):
    parser = _get_parser()
    args = vars(parser.parse_args(args))

    # Do a bit of reformatting of the difficult and pooling args
    args['hard_winners'], args['hard_losers'] = args.pop('difficult')
    (args['win_pool_multiplier'],
     args['loss_pool_multiplier']) = args.pop('pool_multiplier')

    # Always save the dataset when being run from script.
    args['save'] = True

    dataset = build_dataset(**args)

    print(dataset.get_dataset_path())

if __name__ == '__main__':
    main()
