"""
logging_config.py — one place to configure application logging.

Privacy rule: never log raw images, file contents or free-text patient details.
Log internal IDs, hashes (truncated), status codes and timings only.
"""
import logging
import logging.handlers

import config

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("mediscan")
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            config.LOG_DIR / "mediscan.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        root.warning("File logging unavailable; continuing with console logging only.")
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mediscan.{name}")
