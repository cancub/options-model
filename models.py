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
    def __init__(self, model, means, stds, metadata):
        self._model = model
        self._means = means
        self._stds = stds
        self._min_profit = metadata['min_profit']
        self._max_margin = metadata['max_margin']
        self.feature_order = metadata['feature_order']
        self.percentiles = metadata['percentiles']

    @classmethod
    def from_tarball(cls, ticker, id):

        model_fpath = os.path.join(
            config.ML_MODELS_DIR, ticker, '{}.tar'.format(id))

        with tempfile.TemporaryDirectory(prefix='options_model-') as tmpdir:

            tarball_files = utils.extract_and_get_file_list(model_fpath, tmpdir)

            # Make sure all the files we need are there
            for file in ['checkpoint/', 'metadata', 'means', 'variances']:
                assert(file in tarball_files)

            # Load the model
            model = keras.models.load_model(
                os.path.join(tmpdir, 'checkpoint'), compile=False)

            with open(os.path.join(tmpdir, 'metadata'), 'r') as MF:
                metadata = json.load(MF)

            # Load the stats into the metadata
            means = pd.read_pickle(os.path.join(tmpdir, 'means'))
            stds = pd.read_pickle(os.path.join(tmpdir, 'variances')).pow(1/2)

        model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True))

        return cls(model, means, stds, metadata)

    def get_normalized_X_df(self, trades_df):
        return ((trades_df[self.feature_order]
                    - self._means[self.feature_order]) /
                        self._stds[self.feature_order]).values

    def predict_from_normalized_numpy(self, X_arr):
        return self._model.predict(X_arr)[:, 0]

    def predict_from_dataframe(self, trades_df):
        # Normalize that data, make sure its in the right order and return the
        # predictions
        return self.predict_from_normalized_numpy(
                    self.get_normalized_X_df(trades_df))

    def insert_predictions(self, trades_df):
        trades_df.insert(
            0, 'confidence', self.predict_from_dataframe(trades_df))

    def _get_statistics(self, Y_actual, Y_pred):
        pred_trues = Y_pred >= 0
        true_pos = pred_trues & Y_actual
        precision = true_pos.sum() / pred_trues.sum()
        recall = true_pos.sum() / Y_actual.sum()
        f1_score = 2 * precision * recall / (precision + recall)
        return precision, recall, f1_score

    def get_statistics_from_norm_numpy(self, X_arr, Y_actual):
        return self._get_statistics(
            Y_actual, self.predict_from_normalized_numpy(X_arr))

    def get_statistics_from_dataframe(self, trades_df):
        Y_actual = (trades_df.max_profit >= self._min_profit).values
        X_arr = self.get_normalized_X_df(trades_df)
        return self.get_statistics_from_norm_numpy(X_arr, Y_actual)

    def get_statistics_from_batched_tf_dataset(self, dataset):
        test_Xs = []
        test_Ys = []
        for x, y in dataset.as_numpy_iterator():
            test_Xs.append(x)
            test_Ys.append(y)
        test_Xs = np.concatenate(test_Xs)
        test_Ys = np.concatenate(test_Ys)
        return self.get_statistics_from_norm_numpy(test_Xs, test_Ys)

def get_model_details(ticker):
    model_dir = os.path.join(config.ML_MODELS_DIR, ticker)
    ids = []
    models_meta = {
        'max_margin': [],
        'min_profit': [],
        'feature_count': [],
        'accuracy': [],
        'loss': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    for fname in (f for f in os.listdir(model_dir) if f.endswith('.tar')):
        fpath = os.path.join(model_dir, fname)

        ids.append(fname.split('.tar')[0])

        # Get the metadata
        meta = json.loads(
            subprocess.check_output(['tar', '-xOf', fpath, 'metadata']))

        for k in models_meta.keys():
            if k == 'feature_count':
                models_meta[k].append(len(meta['feature_order']))
            else:
                models_meta[k].append(meta[k])

    models_meta['id'] = ids

    return pd.DataFrame(
        models_meta).sort_values(by='id').reset_index(drop=True)

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
    return OptionsModel.from_tarball(
        ticker, good_models.loc[good_models.loss.argmin(), 'id'])
