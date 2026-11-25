"""dashboard de entrenamiento y contrato de eventos independiente del modelo"""

from .dashboard import TrainingDashboard
from .events import TrainingEvent

__all__ = ["TrainingDashboard", "TrainingEvent"]
