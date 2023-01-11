"""Utilities and tools for tracking runs with Mlflow."""

import sys
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import colorstr

LOGGER = logging.getLogger(__name__)


try:
    import mlflow
    assert hasattr(mlflow, "__version__")
except (ImportError, AssertionError):
    mlflow = None

from collections.abc import MutableMapping


class MlflowLogger:
    def __init__(self, opt: Namespace) -> None:
        prefix = colorstr("Mlflow: ")
        try:
            mlflow.set_experiment("YoloV5 - Pragas")
            self.mlflow, self.mlflow_active_run = mlflow, None if not mlflow else mlflow.start_run(run_name=opt.name)
            if self.mlflow_active_run is not None:
                self.run_id = self.mlflow_active_run.info.run_id
                LOGGER.info(f"{prefix}Using run_id({self.run_id})")
                self.setup(opt)
        except Exception as err:
            LOGGER.error(f"{prefix}Failing init - {repr(err)}")
            LOGGER.warning(f"{prefix}Continuining without Mlflow")
            self.mlflow = None
            self.mlflow_active_run = None

    def setup(self, opt: Namespace) -> None:
        if opt.weights is not None and str(opt.weights).strip() != "":
            model_name = Path(opt.weights).stem
        else:
            model_name = "yolov5"
        self.model_name = model_name
        self.weights = Path(opt.weights)
        self.client = mlflow.tracking.MlflowClient()
        self.log_params(vars(opt))

    @staticmethod
    def _format_params(params_dict, parent_key="", sep="/"):
        items = []
        for key, value in params_dict.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(MlflowLogger._format_params(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def log_artifacts(self, artifact: Path, relpath: str = None) -> None:
        if not isinstance(artifact, Path):
            artifact = Path(artifact)
        if artifact.is_dir():
            self.mlflow.log_artifacts(f"{artifact.resolve()}/", artifact_path=str(artifact.stem))
        else:
            self.mlflow.log_artifact(str(artifact.resolve()), artifact_path=relpath)

    def log_model(self, model_path: Path, model_name: str = None) -> None:
        self.mlflow.pyfunc.log_model(artifact_path=self.model_name if model_name is None else model_name,
                                     artifacts={"model_path": str(model_path.resolve())},
                                     python_model=self.mlflow.pyfunc.PythonModel())