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
