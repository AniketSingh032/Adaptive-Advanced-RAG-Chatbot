"""
Logging Configuration Module.

This module provides a centralized logging configuration system for the application.
It handles log file creation, formatting, and logger management.

Example:
    from config.logger import logger
    
    logger.info("Application started")
    logger.error("An error occurred", exc_info=True)
"""

import os
import logging

class Logger:
    """
    A class for setting up and managing a logger.
    """
    def __init__(self, logger_name: str = None, log_format: str = None, datefmt: str = None, level: logging = logging.INFO, filename: str = None):
        """
        Initialize a Logger object.

        Args:
            logger_name (str, optional): The name of the logger. Defaults to None.
            log_format (str, optional): The format for log messages. Defaults to None.
            datefmt (str, optional): The date format for log messages. Defaults to None.
            level (logging, optional): The logging level. Defaults to logging.INFO.
            filename (str): The file where logs will be written. Must be specified.

        Raises:
            ValueError: If the filename is not specified.
        """
        if filename is None:
            raise ValueError("Filename must be specified")
 
        self.log_format = log_format
        self.datefmt = datefmt
        self.level = level
        self.filename = filename
        self.name = logger_name
        self._logger = self._setup_logger()
 
    def _setup_logger(self):
        """
        Set up a logger with the specified parameters.

        Returns:
            logging.Logger: The configured logger instance.
        """
        logging.basicConfig(
            format=self.log_format,
            datefmt=self.datefmt,
            level=self.level,
            filename=self.filename,
        )
 
        return logging.getLogger(self.name)
 
    @property
    def logger(self) -> logging.Logger:
        """
        Get the logger object.

        Returns:
            logging.Logger: The logger instance.
        """
        return self._logger
 
    @logger.setter
    def logger(self, logger):
        """
        Set the logger object.

        Args:
            logger (logging.Logger): The logger instance to set.
        """
        self._logger = logger
 
    def add_handler(self, handler):
        """
        Add a handler to the logger.

        Args:
            handler (logging.Handler): The handler to add to the logger.
        """
        self.logger.addHandler(handler)
 
    def add_filter(self, filter):
        """
        Add a filter to the logger.

        Args:
            filter (logging.Filter): The filter to add to the logger.
        """
        self.logger.addFilter(filter)
 
 
def get_logger():
    """
    Get a logger configured with the settings from the `setup_logger` function.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = setup_logger()
    return logger
 
 
def setup_logger() -> Logger:
    """
    Set up a logger with default settings.

    Returns:
        Logger: The Logger instance with the configured settings.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
   
    log_file = os.path.join(log_dir, 'app.log')
    logger_obj = Logger(
        logger_name=__name__,
        log_format="%(asctime)s [%(levelname)s] '%(pathname)s' %(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file
    )
    
    logger = logger_obj.logger
    return logger
 
logger = get_logger()