"""Local GraphRL integration: expose SDK types without optional remote telemetry."""
from pathlib import Path
__path__.append(str(Path(__file__).resolve().parent / 'agentflow'))
from .types import *
from .server import AgentFlowServer
from .logging import configure_logger
__version__ = '0.1.2'
