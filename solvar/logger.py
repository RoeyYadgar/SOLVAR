import csv
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import colorlog

# Try to import CometML - not a strict dependency
try:
    import comet_ml

    COMET_ML_AVAILABLE = True
except ImportError:
    COMET_ML_AVAILABLE = False
    comet_ml = None

if TYPE_CHECKING:
    from comet_ml import Experiment as CometExperiment


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


class MetricsLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        raise NotImplementedError


class NullMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict, step: int):
        pass


class CSVMetricsLogger(MetricsLogger):
    def __init__(self, path):
        self.path = path
        self.fieldnames = []
        self.rows = []
        self.file_handle = open(self.path, "w")
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)

    def log_metrics(self, metrics: dict, step: int):
        metrics = {"step": step, **metrics}

        # New metric key appears → expand header
        new_keys = [k for k in metrics if k not in self.fieldnames]
        if new_keys:
            self.fieldnames += new_keys
            self._rewrite_file()

        # Append row (fill missing columns with "")
        row = {k: metrics.get(k, "") for k in self.fieldnames}
        self._append_row(row)

    def _append_row(self, row):
        self.writer.writerow(row)
        self.file_handle.flush()

    def _rewrite_file(self):
        """Rewrite full CSV when header expands."""
        if not os.path.exists(self.path):
            return

        with open(self.path, newline="") as f:
            reader = list(csv.DictReader(f))

        self.file_handle.close()
        self.file_handle = open(self.path, "w")
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.writer.writerows(reader)
        self.file_handle.flush()

    def __del__(self):
        self.file_handle.close()


class CometMLMetricsLogger(MetricsLogger):
    def __init__(self, experiment: "CometExperiment | None" = None, **comet_kwargs):
        """Initialize CometML logger.

        Args:
            experiment: Optional existing CometML Experiment object. If None, creates a
                new one.
            **comet_kwargs: Additional arguments passed to comet_ml.Experiment() if
                experiment is None.
                Common kwargs include: project_name, workspace, api_key, etc.
        """

        self.experiment: "CometExperiment | None"
        if not COMET_ML_AVAILABLE:
            logging.debug(f"CometML is not available. {type(self).__name__} will not log metrics.")
            self.experiment = None
            return

        if experiment is not None:
            self.experiment = experiment
        elif self._should_start_new_experiment(comet_kwargs):
            self.experiment = comet_ml.start(**comet_kwargs)
        else:
            logging.debug(
                "No existing CometML experiment found and no new experiment "
                f"parameters provided. {type(self).__name__} will not log metrics."
            )
            self.experiment = None

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to CometML when available."""
        if self.experiment is not None:
            self.experiment.log_metrics(metrics, step=step)

    def _should_start_new_experiment(self, comet_kwargs) -> bool:
        """Determine whether to start a new CometML experiment based on provided kwargs."""

        if os.environ.get("COMET_EXPERIMENT_KEY"):
            # If COMET_EXPERIMENT_KEY is set externally, we assume the user wants to use an existing experiment.
            # in this case comet_ml.start() will automatically use the existing experiment.
            return True

        if "experiment_key" in comet_kwargs.keys() or "project_name" in comet_kwargs.keys():
            # If experiment_key or project_name is explicitly provided in comet_kwargs
            # we should start a new experiment.
            return True

        return False


class CompositeMetricsLogger(MetricsLogger):
    def __init__(self, *metrics_loggers: MetricsLogger):
        self.metrics_loggers = list(metrics_loggers)

    def log_metrics(self, metrics: dict, step: int):
        for metrics_logger in self.metrics_loggers:
            metrics_logger.log_metrics(metrics, step)


def init_metrics_logger(output_path: str | pathlib.Path) -> MetricsLogger:
    return CompositeMetricsLogger(
        CSVMetricsLogger(output_path),
        CometMLMetricsLogger(),
    )
