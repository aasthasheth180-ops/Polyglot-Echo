# backend/logger.py
"""
Structured logging system for Polyglot Echo.
Provides context-aware logging with session tracking and detailed error traces.
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import json

# ── Configure Root Logger ──────────────────────────────────────
class ContextFilter(logging.Filter):
    """Add context variables to log records."""
    def __init__(self):
        super().__init__()
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set context variables (e.g., session_id, user_id)."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables."""
        self.context.clear()
    
    def filter(self, record):
        """Attach context to every log record."""
        record.context = self.context
        return True


# Global context filter instance
_context_filter = ContextFilter()


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured output."""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": _context_filter.context,
        }
        
        # Add file/line info
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str)


def setup_logging(level=logging.INFO):
    """Initialize structured logging."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    console_handler.addFilter(_context_filter)
    logger.addHandler(console_handler)
    
    return logger


def set_context(**kwargs):
    """Set global context variables (session_id, user_id, etc.)."""
    _context_filter.set_context(**kwargs)


def clear_context():
    """Clear all context variables."""
    _context_filter.clear_context()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with context awareness."""
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function entry/exit with timing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"[ENTER] {func.__name__}()", extra={"kwargs": str(kwargs)[:100]})
        
        import time
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"[EXIT] {func.__name__}() completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"[ERROR] {func.__name__}() failed after {elapsed:.3f}s: {str(e)}",
                exc_info=True
            )
            raise
    
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function entry/exit with timing."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"[ENTER] {func.__name__}(async)", extra={"kwargs": str(kwargs)[:100]})
        
        import time
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"[EXIT] {func.__name__}(async) completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"[ERROR] {func.__name__}(async) failed after {elapsed:.3f}s: {str(e)}",
                exc_info=True
            )
            raise
    
    return wrapper


# Initialize on module load
setup_logging()
logger = get_logger("polyglot_echo")