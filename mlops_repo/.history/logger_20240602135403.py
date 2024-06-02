import logging
import sys
from pathlib import Path
from logging.config import dictConfig
from rich.logging import RichHandler

# Define the directory for log files
LOGS_DIR = Path('./logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration dictionary
logging_config = {
    "version": 1,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler"
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "info.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "error.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.DEBUG,
        "propagate": True,
    },
}

# Apply logging configuration
dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Test logging levels again
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")

# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
