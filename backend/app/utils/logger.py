"""
Logging configuration for the AI Video Search system
"""
import sys
from pathlib import Path
from loguru import logger

from app.config import settings


def setup_logger():
    """Setup application logging"""
    
    # Remove default logger
    logger.remove()
    
    # Create logs directory
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console logger
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # File logger
    logger.add(
        settings.LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    # Error file logger
    error_log_file = str(Path(settings.LOG_FILE).parent / "error.log")
    logger.add(
        error_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info("Logger setup completed")


# Setup logger on import
setup_logger()
