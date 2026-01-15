import logging

import colorlog


def setup_logger(level: int = logging.INFO) -> None:
    """Set up a logger with custom formatting.

    Args:
        level: Logging level (default: logging.INFO)
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s.%(msecs)03d %(levelname)s (%(module)s) - (%(funcName)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
