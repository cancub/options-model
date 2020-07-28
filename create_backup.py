from datetime import datetime
from glob import glob
from subprocess import check_call
import os
import subprocess

from config import STORAGE_DIR, BACKUPS_DIR

filepath = os.path.join(BACKUPS_DIR, '{}.tar.gz'.format(datetime.now().date()))

if not os.path.exists(BACKUPS_DIR):
    os.makedirs(BACKUPS_DIR)

check_call(
    ['tar', '-C', STORAGE_DIR,'-czf', filepath] + \
        [f.split('/')[1] for f in glob('{}/*'.format(STORAGE_DIR))]
)