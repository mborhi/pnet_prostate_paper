import sys
from os.path import join, dirname, realpath, exists
from os import makedirs
current_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, dirname(current_dir))
from config_path import PLOTS_PATH

saving_dir = join(PLOTS_PATH, 'suplmentary')
if not exists(saving_dir):
    makedirs(saving_dir)