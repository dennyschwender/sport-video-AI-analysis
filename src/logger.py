import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """Centralized logging configuration."""
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "floorball_llm", log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
        """Get or create logger instance."""
        if cls._instance is None:
            cls._instance = cls._setup_logger(name, log_file, level)
        return cls._instance
    
    @classmethod
    def _setup_logger(cls, name: str, log_file: Optional[str], level: str) -> logging.Logger:
        """Configure logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []  # Clear existing handlers
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            # Resolve log file path relative to project root (where app.py is)
            # This ensures logs/ folder is always in the project directory
            project_root = Path(__file__).resolve().parent.parent
            log_path = project_root / log_file
            
            # Create logs directory if it doesn't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_path}")
        
        return logger


def log_llm_call(logger: logging.Logger, backend: str, model: str, input_length: int, 
                 processing_ms: int, cost_usd: float, events_found: int):
    """Log structured LLM call metrics."""
    logger.info(
        f"LLM Call - Backend: {backend}, Model: {model}, "
        f"Input: {input_length}chars, Time: {processing_ms}ms, "
        f"Cost: ${cost_usd:.6f}, Events: {events_found}"
    )


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict):
    """Log error with contextual information."""
    logger.error(
        f"Error: {type(error).__name__}: {str(error)}\n"
        f"Context: {context}",
        exc_info=True
    )
