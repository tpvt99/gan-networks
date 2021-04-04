"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""

from .config import RESULTS_DIR_PATH
from .config import PROGRESS_FILE_NAME, WEIGHTS_FOLDER_NAME, CONFIG_FILE_NAME, CHECKPOINT_FOLDER_NAME
import time
import os.path as osp
import os
import atexit
from .utils import colorize, convert_json
import shutil
import json
import joblib
import tensorflow as tf


class Logger():
    '''
    Logger to save the models, information and logging statistics
    Steps to save are:
    1. logger.save_config(locals()) to store all the configuration (should sit a first line in training script)
    2. logger.setup_tf_saver(tf_model) to setup the tf model
    3. For epoch:
            for mini-batch:
                logger.store() to store each stats of batch ( use or not use is fine)
            logger.save_state() to save the tf model with best accuracy
            logger.log_tabular() to calculate values stored by logger.store()
                                and print the neccesary info and save
                                into a progress temp dict
            logger.dump_tabular() to flush the progress temp dict into file
    '''
    def __init__(self, data_dir=None, exp_name=None, output_fname = None):
        '''
        Initialize the output directory
        :param data_dir: the root folder to contain running results (Currently, dont' use because we have ray)
        :param exp_name: the current algorithm
        :param output_fname: the logging output to contain progress statistics
        '''

        #1. Set up exp_name, suffix, output_dir, output_file
        # exp_name
        assert exp_name is not None, "You must provide the exp_name"
        self.exp_name = exp_name

        # suffix + output_dir
        data_dir = data_dir or RESULTS_DIR_PATH
        self.output_dir = self.setup_output_dir(data_dir=data_dir)

        if osp.exists(self.output_dir):
            print("Warning: {0} exists. Storing data into this folder anyway.".format(self.output_dir))
        else:
            os.makedirs(self.output_dir)
            print("Creating {0}. Storing the progress file in this folder.".format(self.output_dir))

        # output_file
        output_fname = output_fname or PROGRESS_FILE_NAME
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))

        #2. Set up log headers
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {} # Store value of current epoch

        #3. Set up tensorboard
        train_log_dir = osp.join(self.output_dir, 'logs', 'gradient_tape', 'train')
        test_log_dir = osp.join(self.output_dir, 'logs', 'gradient_tape', 'test')
        self.func_log_dir = osp.join(self.output_dir, 'logs', 'func')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.func_summary_writer = tf.summary.create_file_writer(self.func_log_dir)

        #4. Set up folder for outputing generated images
        self.image_dir = osp.join(self.output_dir, 'images')

    def setup_output_dir(self, data_dir=None, datestamp=True):
        '''
        We follow spinningup logging name
        Config the output_dir = data_dir/exp_name/[outer_prefix]exp_name/[inner_prefix]exp_name

        :param data_dir:
        :param datestamp:
        :return:
        '''
        ymd_time = time.strftime("%Y-%m-%d") if datestamp else ""
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S") if datestamp else ""

        #1. exp_name
        data_dir = osp.join(data_dir, self.exp_name)

        #2. [outer_prefix]_exp_name
        relpath = ''.join([ymd_time, '_', self.exp_name])
        data_dir = osp.join(data_dir, relpath)

        #3. [inner_prefix]exp_name
        subfolder = ''.join([hms_time, '_', self.exp_name])

        data_dir = osp.join(data_dir, subfolder)

        return data_dir

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = Logger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(self.output_dir, CONFIG_FILE_NAME), "w") as f:
            f.write(output)


    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration (epoch).
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def dump_tabular(self):
        '''
        Write all of the diagnostics from the current iteration.
        Call this once only after  finishing running an epoch
        Writes both to stdout, and to the output file.
        :return:
        '''
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        max_val_len = 15
        print_fmt = f'| {{:>{max_key_len}}} | {{:>{max_val_len}}} |'
        n_slashes = 22 + max_key_len
        print("*" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else f"{val}"
            print(print_fmt.format(key, valstr))
            vals.append(val)
        print('*' * n_slashes, flush=True)
        if self.output_file is not None:
            max_key_len = max(key_lens) # Reassign for flushing to file
            if self.first_row:
                key_lines = [f'{key:{max_key_len+2}.{max_key_len}}' for key in self.log_headers]
                self.output_file.write(''.join(key_lines)+'\n')
            val_lines = [f'{val:<{max_key_len+2}.5g}' if hasattr(val,'__float__') else f'{val:<{max_key_len+2}}' for val in vals] # add 2 to easy to see
            self.output_file.write(''.join(val_lines)+'\n')
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def save_state(self, itr=None):
        self._tf_simple_save(itr)

    def setup_tf_saver(self, what_to_save):
        """
        Set up easy model saving for a single Tensorflow model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any Tensorflow model or serializable object containing
                Tensorflow models.
        """
        self.tf_saver_elements = what_to_save

    def _tf_simple_save(self, itr = None):
        """
        Uses simple_save to save a trained model, plus info to make it easy
        to associated tensors to variables after restore.
        """
        assert hasattr(self, 'tf_saver_elements'), \
            "First have to setup with self.setup_tf_save"
        fpath = osp.join(self.output_dir, WEIGHTS_FOLDER_NAME)
        fpath = osp.join(fpath, f'tf_save{itr}' if itr is not None else f'tf_save')
        if osp.exists(fpath):
            # tf save model inside directory. If directory exists, simply delete this
            shutil.rmtree(fpath)
        self.tf_saver_elements.save(fpath)

    def tf_board_save_scaler(self, epoch, val, name, type):
        '''
        To save parameters to TensorBoard
        :param epoch:
        :param val: value of scaler to export
        :param type: 'train' or 'test'
        :param name: name of scaler to export
        :return:
        '''

        if type == 'train':
            with self.train_summary_writer.as_default():
                tf.summary.scalar(name, val, step=epoch)

        if type == 'test':
            with self.test_summary_writer.as_default():
                tf.summary.scalar(name, val, step=epoch)

    def tf_board_save_graph(self):
        with self.func_summary_writer.as_default():
            tf.summary.trace_export(name="my_func_trace", step=0)

    def save_images(self, plt, name):
        if not osp.exists(self.image_dir):
            os.makedirs(self.image_dir)
        plt.savefig(osp.join(self.image_dir, name))

class CheckpointLogger(Logger):
    def __init__(self, data_dir=None, exp_name=None, output_fname = None):
        super().__init__(data_dir, exp_name, output_fname)

    def setup_tf_checkpoint(self, checkpoint, max_to_keep):
        """
        Set up easy model saving for a single Tensorflow model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any Tensorflow model or serializable object containing
                Tensorflow models.
        """
        self.tf_checkpoint = checkpoint
        checkpoint_path = osp.join(self.output_dir, CHECKPOINT_FOLDER_NAME)
        self.tf_checkpoint_manager = tf.train.CheckpointManager(self.tf_checkpoint,
                                                                checkpoint_path, max_to_keep=max_to_keep)

        if self.tf_checkpoint_manager.latest_checkpoint:
            self.tf_checkpoint.restore(self.tf_checkpoint_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def save_checkpoint(self):
        assert hasattr(self, 'tf_checkpoint_manager'), \
            "First have to setup with self.setup_tf_checkpoint"
        self.tf_checkpoint_manager.save()

