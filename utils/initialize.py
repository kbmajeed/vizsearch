import datetime
import logging
import os
import yaml

from box import Box


def load_config(config_main_path: str = '../config/') -> Box:
    """
    Create all the configuration files and make available throughout project
    :param config_main_path: directory containing config files
    :return: Box object containing config variables
    """

    config_files = os.listdir(config_main_path)
    config_files = [i_config_file for i_config_file in config_files if i_config_file.endswith('.yaml')]

    config_f = {}

    for i_config_file in range(len(config_files)):
        config_file_name = config_files[i_config_file]
        i_config_file_path = os.path.join(config_main_path, config_file_name)

        with open(i_config_file_path, 'r') as yaml_file:
            logging.info(f"Loading configuration file: {config_file_name}")
            config_f[config_file_name.split('.')[0]] = yaml.safe_load(yaml_file)

    config_f = Box(config_f)

    return config_f


config = load_config()


def load_logging(logging_main_path: str = '../logs/') -> None:
    """
    Setup logging systems accessible for entire project
    :param logging_main_path:
    :return:
    """

    log_full_path = os.path.join(logging_main_path, 'vizsearch.log')
    logging.basicConfig(
        filename=log_full_path,
        filemode='w',
        encoding='utf-8',
        format="%(asctime)s - %(levelname)s - [%(filename)s: %(funcName)22s(): %(lineno)s]  -  %(message)s",
        level=logging.INFO,
        force=True,
    )
    stream_handle = logging.StreamHandler()
    stream_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s: %(funcName)22s(): %(lineno)s]  -  %(message)s")
    stream_handle.setFormatter(stream_format)
    logging.getLogger().addHandler(stream_handle)

    logging.info('==='*22)
    logging.info(f'*** Vizsearch Log *** | {datetime.datetime.now()}')
    logging.info('==='*22)

    return None
