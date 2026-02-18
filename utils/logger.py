import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_file: Path, level_name: str = "INFO"):
    """Setup logging to console and file with rotation.
    
    Args:
        log_file: Path to the log file.
        level_name: Logging level name (DEBUG, INFO, WARNING, ERROR).
    """
    # Map string level to logging constants
    level = getattr(logging, level_name.upper(), logging.INFO)
    
    # Create directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Format for logging
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    
    # Rotating File handler (5MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5 * 1024 * 1024, 
        backupCount=5, 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Level: {level_name}, Output: {log_file}")

def get_logger(name: str):
    """Get a named logger."""
    return logging.getLogger(name)
