from   datetime import datetime
import errno
import os
import pandas as pd
import subprocess
import tempfile

import config
import utils

NOW = datetime.now().date()
backup_path = utils.get_last_backup_path()

with tempfile.TemporaryDirectory() as tmpdir:
    files = utils.extract_and_get_file_list(backup_path, tmpdir)

    for f in files:
        ticker = os.path.splitext(f)[0]
        # Load the DataFrame
        df_path = os.path.join(tmpdir, f)
        df = pd.read_pickle(df_path)

        # Spin off the old expiries

        # Get a list of all of the expiries that are done
        expiries = df.index.get_level_values(level=1)

        # Get the selectors for the DataFrame for expiries before and after
        # today
        old = expiries.map(lambda x: datetime.strptime(
            x, '%Y-%m-%d').date() <= NOW).values
        new = expiries.map(lambda x: datetime.strptime(
            x, '%Y-%m-%d').date() > NOW).values

        before_df = df[old]

        if len(before_df) > 0:
            ticker_dir = os.path.join(config.EXPIRIES_DIR, ticker)
            try:
                os.makedirs(ticker_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            # Walk through the old expiries and save them in their own pickles
            for old_exp in before_df.index.unique(level=1).values:
                # It might be that we've picked up data for an expiry that has
                # already passed. Just ignore these cases and let the data die
                exp_path = os.path.join(ticker_dir, '{}.bz2'.format(old_exp))
                if os.path.exists(exp_path):
                    continue
                exp_df = before_df.xs(old_exp, level=1, drop_level=False)
                exp_df.to_pickle()

        # Re-add the remaining, unexpired data to the backup to be saved once
        # more
        df[new].to_pickle(df_path)

    # Make the all-unexpired backup
    subprocess.check_call(
        ['tar', '-C', tmpdir, '-cf', backup_path + '.tmp'] + files)
