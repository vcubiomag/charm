import configparser
import json

from charm.logging import logger


def read_ini(config_file):
    """Read an .ini file and return its content as a nested dictionary.

    Values are parsed as JSON, so the type follows the literal:
        int = 1
        float = 1.2
        bool = true/false (lower case)
        str = "string" (double quotes)
        list = [1, 2, 3]
        dict = {"a": 1, "b": 2} (keys must be strings)

    Parameters
    -------------
    config_file: str
        Name of configuration file

    Returns
    ---------
    config: dict
        Nested dictionary with parsed values for each section
    """
    logger.info("Reading settings from " + str(config_file))
    cfg = configparser.ConfigParser()
    fn = cfg.read(config_file)
    if not len(fn):
        logger.error("error when reading " + str(config_file))
        raise OSError("error when reading config file")

    config = {}
    for section in cfg.sections():
        config[section] = {}
        for key in cfg[section]:
            try:
                config[section][key] = json.loads(cfg[section][key])
            except json.decoder.JSONDecodeError:
                logger.error(f"error when formatting {section} {key}={cfg[section][key]}")
                raise OSError("error when parsing config file")

    return config
