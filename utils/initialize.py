import os
import logging

import yaml

from box import Box


def setup_config_files(config_main_path: str = '../config/') -> Box:
    """
    Create all the configuration files and make available throughout project
    :param config_main_path:
    :return: Box object
    """

    config_files = os.listdir(config_main_path)
    config_files = [i_config_file for i_config_file in config_files if i_config_file.endswith('.yaml')]

    config = {}

    for i_config_file in range(len(config_files)):
        config_file_name = config_files[i_config_file]
        i_config_file_path = os.path.join(config_main_path, config_file_name)

        with open(i_config_file_path, 'r') as yaml_file:
            logging.info(f"Loading configuration file: {config_file_name}")
            config[config_file_name.split('.')[0]] = yaml.safe_load(yaml_file)

    config = Box(config)

    return config


def setup_logging(logging_main_path: 'str' = '../logs/') -> None:
    """
    Setup logging systems accessible for entire project
    :param logging_main_path:
    :return:
    """

    log_full_path = os.path.join(logging_main_path, 'vizsearch.log')
    if not os.path.exists(log_full_path):
        logging.basicConfig(
            filename=log_full_path,
            filemode='w',
            encoding='utf-8',
            format='%(asctime)s %(message)s',
            level=logging.INFO)
        logging.info('==='*15)
    else:
        pass

    return None


if __name__ == '__main__':
    config_dir = '../config/'
    config = setup_config_files(config_dir)
