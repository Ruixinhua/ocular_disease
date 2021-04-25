import logging
import logging.config
import os

from utils.tools import read_json


def setup_logging(save_dir, log_config="", default_level=logging.INFO):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_config):
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        # print(save_dir / "info.log")
        # print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            filename=str(save_dir / "info.log"), force=True)
