from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

__all__: list[str] = ["PropagationType", "PropagationOptions", "ODEVectorizedOptions"]


class PropagationType(Enum):
    """
    Enum to specify the propagation type

    Attributes
        ballistic (int): ballistic trajectory
        ode (int): ode trajectory
        linear (int): linear restoring force trajectory
        ode_vectorized (int): vectorized ode trajectory
    """

    ballistic = auto()
    ode = auto()
    linear = auto()
    ode_vectorized = auto()


@dataclass
class ODEVectorizedOptions:
    n_steps: int
    n_save: int


@dataclass
class PropagationOptions:
    """
    Dataclass to hold Propagation options

    Attributes
        n_cores (int): # cores used to solve ODE trajectories
        verbose (bool): enable verbose logging of joblib and others
    """

    n_cores: int = 6
    verbose: bool = True
    section_options: dict[str, Any] = field(default_factory=dict)
