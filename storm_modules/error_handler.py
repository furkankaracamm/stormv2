"""
Global Exception Handler - Centralized Error Tracking

Features:
- Centralized error logging
- Context-aware exception handling
- Safe execution decorators
"""

import sys
import traceback
import logging
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime

class GlobalExceptionHandler:
    def __init__(self, log_file: str = "storm_errors.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Configure error logger."""
        self.logger = logging.getLogger("STORM_Governance")
        self.logger.setLevel(logging.ERROR)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)

    def log_error(self, e: Exception, context: str = "General", severity: str = "ERROR"):
        """Log an error with context and stack trace."""
        tb = traceback.format_exc()
        error_msg = f"[{context}] {type(e).__name__}: {str(e)}\nStack Trace:\n{tb}"
        
        if severity == "CRITICAL":
            self.logger.critical(error_msg)
            print(f"\n[CRITICAL ERROR] {context}: {e} (See log for details)")
        else:
            self.logger.error(error_msg)
            print(f"\n[ERROR] {context}: {e}")

    def safe_execute(self, context: str = "Unknown"):
        """Decorator for safe execution with error logging."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.log_error(e, context)
                    return None  # Fail safe
            return wrapper
        return decorator

    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Hook for sys.excepthook."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.critical(
            "Uncaught Exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        print(f"[FATAL] Uncaught exception: {exc_value}")

# Global instance
error_handler = GlobalExceptionHandler()
