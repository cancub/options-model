from   datetime import datetime
from   glob import glob
import pandas as pd
from   subprocess import check_call, check_output
from   tempfile import TemporaryDirectory
import os

from   config import STORAGE_DIR, BACKUPS_DIR
import utils

# Find the last backup tarball
last_backup = utils.get_last_backup_path()

# Get the list of tickers that were generated today
todays_tickers = os.listdir(STORAGE_DIR)

# Extract this tarball into a staging directory so that we can append to the
# existing DataFrames for each ticker
with TemporaryDirectory() as tmpdirname:
    print('unpacking ' + last_backup)
    check_call(['tar', '-C', tmpdirname, '-xf', last_backup])

    # Go through each of the tickers that were parsed today
    for tik in os.listdir(STORAGE_DIR):
        compressed_path = os.path.join(tmpdirname, tik + '.bz2')

        print('Loading today\'s data for {}'.format(tik))
        # Load in the data generated today
        new_df = pd.read_pickle(os.path.join(STORAGE_DIR, tik))

        print('Appending to existing data')
        # Load in the stored data for this ticker and append the new data to
        # this pickle
        try:
            df = pd.concat((pd.read_pickle(compressed_path), new_df))
        except FileNotFoundError:
            df = new_df

        print('Repickling')
        # (Over)write the compressed pickle with the new concatenated data
        df.to_pickle(compressed_path)

    import pdb; pdb.set_trace()

    # Get the name for the new tarball of pickles
    filepath = os.path.join(
        BACKUPS_DIR, '{}.tar'.format(datetime.now().date()))

    # Build the latest tarball
    check_call(
        ['tar', '-C', tmpdirname, '-cf', filepath] +
            [os.path.basename(f) for f in glob('{}/*'.format(tmpdirname))]
    )
