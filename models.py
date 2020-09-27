import json
import numpy as np
import os
import pandas as pd
import subprocess
import tempfile

import config
import utils

from tensorflow import keras

class OptionsModel(object):
    def __init__(self, ticker, filename):
        self.ticker   = ticker
        self.filename = filename

        model_fpath = os.path.join(config.ML_MODELS_DIR, ticker, filename)

        with tempfile.TemporaryDirectory(prefix='options_model-') as tmpdir:

            tarball_files = utils.extract_and_get_file_list(model_fpath, tmpdir)

            # Make sure all the files we need are there
            for file in ['checkpoint/', 'metadata', 'means', 'variances']:
                assert(file in tarball_files)

            # Load the model
            self._model = keras.models.load_model(
                os.path.join(tmpdir, 'checkpoint'), compile=False)

            # Load the stats
            self._means = pd.read_pickle(os.path.join(tmpdir, 'means'))
            self._stds = pd.read_pickle(
                os.path.join(tmpdir, 'variances')).pow(1/2)

            with open(os.path.join(tmpdir, 'metadata'), 'r') as MF:
                self._feature_order = json.load(MF)['feature_order']

        self._model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True))

    def predict(self, trades_df):
        # Normalize that data, make sure its in the right order and return the
        # predictions
        return self._model.predict(
            ((trades_df[self._feature_order]
                - self._means[self._feature_order]) /
                    self._stds[self._feature_order]).values
        )

    def insert_predictions(self, trades_df):
        trades_df.insert(0, 'confidence', self.predict(trades_df))

def get_model_details(ticker):
    model_dir = os.path.join(config.ML_MODELS_DIR, ticker)
    filenames = []
    models_meta = {
        'max_margin': [],
        'min_profit': [],
        'feature_count': [],
        'accuracy': [],
        'loss': []
    }

    for fname in (f for f in os.listdir(model_dir) if f.endswith('.tar')):
        fpath = os.path.join(model_dir, fname)

        filenames.append(fpath)

        # Get the metadata
        meta = json.loads(
            subprocess.check_output(['tar', '-xOf', fpath, 'metadata']))

        for k in models_meta.keys():
            if k == 'feature_count':
                models_meta[k].append(len(meta['feature_order']))
            else:
                models_meta[k].append(meta[k])

    models_meta['filename'] = filenames

    return pd.DataFrame(
        models_meta).sort_values(by='filename').reset_index(drop=True)

def load_best_model(ticker, max_margin=np.inf, min_profit = 0):
    # Get the details of all the stored models
    details = get_model_details(ticker)

    # Only look atthose which satisfy the arguments
    good_models = details[
        (details.max_margin <= max_margin)
            & (details.min_profit >= min_profit)
    ]

    if len(good_models) == 0:
        raise Exception(
            ('No models meet the criteria of margin <= {}, profit threshold '
             '>= {}').format(max_margin, min_profit)
        )

    # Load the model and statistics from the tarball
    return OptionsModel(
        ticker,
        good_models.loc[good_models.loss.argmin(), 'filename']
    )
