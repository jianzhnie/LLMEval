import logging
import sys
from typing import Optional


def init_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
) -> logging.Logger:
    """Initialize and configure a logger.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_file (Optional[str], optional): Path to log file. If None, logs to stdout only.
        log_format (str, optional): The format string for log messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create and add stream handler (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
