import os
import os.path as osp

### Configuration for logging results

RESULTS_DIR_PATH = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'results')
PROGRESS_FILE_NAME = 'progress.csv'
WEIGHTS_FOLDER_NAME = 'tf_model'
CHECKPOINT_FOLDER_NAME = 'tf_checkpoint'
CONFIG_FILE_NAME = 'config.json'