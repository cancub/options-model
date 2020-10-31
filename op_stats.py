import os
import pandas as pd

import config
import utils

def nullify_non_norm(means, variances):
    # First, ensure that any of the columns that should not be normalize or are
    # going to be normalized by other means.
    for c in config.NON_NORM_COLS:
        try:
            means[c] = 0
            variances[c] = 1
        except KeyError:
            pass
    return means, variances

def collect_statistics(trades_df):
    trades_means = trades_df.mean()
    trades_vars = trades_df.var()
    return nullify_non_norm(trades_means, trades_vars)

def pool_stats_from_stats_df(ticker):
    stats_df = pd.read_pickle(os.path.join(
        config.SPREADS_DIR, ticker, 'stats'))

    # Only collect one set of pools and samples (not doubling up for means and
    # variances)
    pools = stats_df.pop('pools').xs('mean', level=1)
    samples = stats_df.pop('samples').xs('mean', level=1)

    means = stats_df.xs('mean', level=1)
    variances = stats_df.xs('variance', level=1)

    new_means = means.multiply(samples, 0).sum() / samples.sum()
    new_variances = variances.multiply(
        samples-pools, 0).sum() / (samples.sum() - pools.sum())

    # Ensure that some columns will not be normalized
    return nullify_non_norm(new_means, new_variances)

def pool_stats_from_expiry(expiry_path):

    # Collect the statistics
    means = []
    variances = []
    sample_sizes = []

    # Get the stats for each DataFrame in the expiry tarball
    for df in utils.spreads_tarballs_to_generator(expiry_path, shuffle=True):
        # Only collect statistics for values that willl be used for training
        columns = df.columns.drop(config.STATS_IGNORE_COLS)

        df_means, df_vars = collect_statistics(df[columns])
        means.append(df_means)
        variances.append(df_vars)
        sample_sizes.append(df.shape[0])

    # Collect the numerators for both means and variances
    pooled_means = None
    pooled_vars = None
    total_samples = sum(sample_sizes)
    for i in range(len(sample_sizes)):
        next_mean = means[i] * sample_sizes[i]
        next_var = variances[i] * (sample_sizes[i] - 1)
        try:
            pooled_means += next_mean
            pooled_vars += next_var
        except TypeError:
            pooled_means = next_mean
            pooled_vars = next_var

    # Divide the result by the respective denominators
    pool_count = len(sample_sizes)
    pooled_means /= total_samples
    pooled_vars /= (total_samples - pool_count)

    # Finally, reset some of the values that should not be changed
    pooled_means, pooled_vars = nullify_non_norm(pooled_means, pooled_vars)

    # Also provide the number of samples with the stats for combining these
    # pooled values with other pooled values later on
    return pooled_means, pooled_vars, total_samples, pool_count
