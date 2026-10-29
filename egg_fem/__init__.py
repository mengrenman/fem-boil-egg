"""FEM simulation of heat transfer during egg boiling.

Solves the axisymmetric transient heat equation:
    ρ(T)·Cp(T)·∂T/∂t = ∇·(κ(T)·∇T)
using the finite element method with scikit-fem.
"""

from .materials import (
    EggMaterial,
    egg_white_properties,
    egg_yolk_properties,
    egg_white_denaturation_properties,
    egg_yolk_denaturation_properties,
    WHITE_CONST,
    YOLK_CONST,
    T_BOILING,
    T_ROOM,
    T_FRIDGE,
    WHITE_DENATURATION_K,
    YOLK_DENATURATION_K,
    YOLK_FULL_SET_K,
    H_BOILING_SHELL,
)
from .geometry import (
    create_circular_mesh,
    create_egg_mesh,
    egg_profile,
)
from .solver import (
    HeatSolver,
    mass_axisym,
    stiffness_axisym,
    robin_lhs_axisym,
    robin_rhs_axisym,
)
from .kinetics import (
    ArrheniusKinetics,
    WHITE_KINETICS,
    YOLK_KINETICS,
    compute_denaturation_history,
)
