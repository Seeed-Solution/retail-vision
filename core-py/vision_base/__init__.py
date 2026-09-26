"""Python orchestration/transport layer of the unified vision base.

Standard library only (spec BASE-1 §5.2.1); no per-frame pixel or tensor
computation lives here.
"""
__version__ = "0.1.0"

from .types import Detection, Event, FrameResult, Hello, LetterboxGeom, StreamSpec
from .runtime_client import RuntimeClient, RuntimeError_, RuntimeGone
from .hooks import AppHooks, ConfigureError, EchoApp, Outgoing, StreamContext
from .apps import ConfigApp
from .shard import Shard
from .supervisor import Supervisor
from .health import HealthServer, build_healthz

__all__ = [
    "LetterboxGeom",
    "Detection",
    "FrameResult",
    "Event",
    "StreamSpec",
    "Hello",
    "RuntimeClient",
    "RuntimeError_",
    "RuntimeGone",
    "AppHooks",
    "ConfigureError",
    "EchoApp",
    "ConfigApp",
    "Outgoing",
    "StreamContext",
    "Shard",
    "Supervisor",
    "HealthServer",
    "build_healthz",
    "__version__",
]
