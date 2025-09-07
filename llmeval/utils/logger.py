import logging
import sys
from pathlib import Path
from typing import Optional, Union


def init_logger(
    name: str,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    console_output: bool = True,
    file_mode: str = 'a',
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    propagate: bool = False,
) -> logging.Logger:
    """Initialize and configure a logger with enhanced features.

    Args:
        name (str): The name of the logger.
        level (Union[int, str], optional): The logging level. Can be int or string.
            Defaults to logging.INFO.
        log_file (Optional[Union[str, Path]], optional): Path to log file.
            If None, logs to stdout only.
        log_format (Optional[str], optional): The format string for log messages.
            If None, uses a default format.
        console_output (bool, optional): Whether to output to console. Defaults to True.
        file_mode (str, optional): File mode for log file ('a' for append, 'w' for write).
            Defaults to 'a'.
        max_bytes (int, optional): Maximum size of log file before rotation. Defaults to 10MB.
        backup_count (int, optional): Number of backup files to keep. Defaults to 5.
        propagate (bool, optional): Whether to propagate messages to parent loggers.
            Defaults to False.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)
    logger.propagate = propagate

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set default format if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create and add stream handler (console output) if enabled
    if console_output:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)

        # Create directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use RotatingFileHandler for log rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(log_path,
                                               mode=file_mode,
                                               maxBytes=max_bytes,
                                               backupCount=backup_count,
                                               encoding='utf-8')
        except ImportError:
            # Fallback to regular FileHandler if RotatingFileHandler is not available
            file_handler = logging.FileHandler(log_path,
                                               mode=file_mode,
                                               encoding='utf-8')

        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: Union[int, str]) -> None:
    """Set the logging level for a logger and all its handlers.

    Args:
        logger (logging.Logger): The logger to configure.
        level (Union[int, str]): The logging level.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
