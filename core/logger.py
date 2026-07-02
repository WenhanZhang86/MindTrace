import logging
import sys
from pathlib import Path


def setup_logging(app_dir: Path) -> logging.Logger:
    logs_dir = app_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    root_logger = logging.getLogger("context_assistant")
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"context_assistant.{name}")
