"""Centralized logging configuration for RepTate."""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_CONFIGURED = False


class _SessionFilter(logging.Filter):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach session ID to log record.

        Args:
            record: Log record to filter

        Returns:
            Always True (all records pass through with session ID attached)
        """
        record.session_id = self._session_id
        return True


def _coerce_level(level: Optional[int | str], fallback: int) -> int:
    if level is None:
        return fallback
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(str(level).upper(), fallback)


def _log_dir_from_qt() -> Optional[Path]:
    try:
        from PySide6.QtCore import QStandardPaths

        path = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        return Path(path)
    except Exception:
        return None


def get_log_dir() -> Path:
    """Determine and create the log directory.

    Uses REPTATE_LOG_DIR environment variable if set, otherwise attempts
    to use Qt's AppDataLocation, falling back to ~/.local/share/RepTate.

    Returns:
        Path to the log directory (created if it doesn't exist)
    """
    env_dir = os.environ.get("REPTATE_LOG_DIR")
    if env_dir:
        path = Path(env_dir)
    else:
        path = _log_dir_from_qt() or (Path.home() / ".local" / "share" / "RepTate")
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(
    name: str = "RepTate",
    level: Optional[int | str] = None,
    console_level: Optional[int | str] = None,
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_error_file: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure centralized logging for RepTate.

    Sets up rotating file handlers for general and error logs, plus optional
    console output. Respects environment variables for fine-grained control.

    Environment Variables:
        REPTATE_LOG_DIR: Override log directory location
        REPTATE_LOG_CONSOLE_LEVEL: Console output level (default: WARNING)
        REPTATE_LOG_FILE_LEVEL: File output level (default: INFO)
        REPTATE_LOG_ERROR_LEVEL: Error file level (default: ERROR)
        REPTATE_LOG_MAX_BYTES: Max bytes per log file (default: 2000000)
        REPTATE_LOG_BACKUP_COUNT: Number of backup files (default: 5)
        REPTATE_SESSION_ID: Session identifier (default: process PID)

    Args:
        name: Logger name (default: "RepTate")
        level: Base logging level (default: INFO)
        console_level: Console handler level (default: WARNING)
        log_dir: Log directory (default: from get_log_dir())
        enable_console: Enable console output handler
        enable_file: Enable general file handler
        enable_error_file: Enable error-specific file handler
        force: Reconfigure even if already configured

    Returns:
        Configured logger instance
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return logging.getLogger(name)

    log_dir = log_dir or get_log_dir()

    level = _coerce_level(level, logging.INFO)
    console_level = _coerce_level(
        console_level, _coerce_level(os.environ.get("REPTATE_LOG_CONSOLE_LEVEL"), logging.WARNING)
    )
    file_level = _coerce_level(os.environ.get("REPTATE_LOG_FILE_LEVEL"), level)
    error_level = _coerce_level(os.environ.get("REPTATE_LOG_ERROR_LEVEL"), logging.ERROR)

    max_bytes = int(os.environ.get("REPTATE_LOG_MAX_BYTES", "2000000"))
    backup_count = int(os.environ.get("REPTATE_LOG_BACKUP_COUNT", "5"))

    session_id = os.environ.get("REPTATE_SESSION_ID", f"{os.getpid()}")
    session_filter = _SessionFilter(session_id)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = "%(asctime)s %(name)s %(levelname)s [%(process)d:%(threadName)s] [%(session_id)s] %(message)s"
    datefmt = "%Y%m%d %H%M%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if enable_file:
        log_file = log_dir / "RepTate.log"
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        fh.addFilter(session_filter)
        logger.addHandler(fh)

    if enable_error_file:
        err_file = log_dir / "RepTate.error.log"
        eh = logging.handlers.RotatingFileHandler(
            err_file, maxBytes=max_bytes, backupCount=backup_count
        )
        eh.setLevel(error_level)
        eh.setFormatter(formatter)
        eh.addFilter(session_filter)
        logger.addHandler(eh)

    if enable_console:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        ch.addFilter(session_filter)
        logger.addHandler(ch)

    logging.captureWarnings(True)
    _CONFIGURED = True
    logger.debug("Logging configured: dir=%s level=%s", log_dir, logging.getLevelName(level))
    return logger
