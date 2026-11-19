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
    mesh_volume,
)
from .solver import (
    HeatSolver,
    solve_condense,
    mass_axisym,
    stiffness_axisym,
    robin_lhs_axisym,
    robin_rhs_axisym,
)
from .doneness import (
    DONENESS_DEFINITIONS,
    MEDIUM_YOLK_FRACTION,
    center_crossing_times,
    crossing_time,
    doneness_times,
)
from .options import (
    black_scholes_call,
    solve_black_scholes_heat_fem,
    fourier_sphere_center_temp,
    ThermalResponse,
    KitchenUncertainty,
    check_similarity_against_fem,
    price_egg_corridor_option,
    egg_culinary_payoff,
    solve_american_egg_option,
)

__all__ = [
    # materials
    "EggMaterial",
    "egg_white_properties",
    "egg_yolk_properties",
    "egg_white_denaturation_properties",
    "egg_yolk_denaturation_properties",
    "WHITE_CONST",
    "YOLK_CONST",
    "T_BOILING",
    "T_ROOM",
    "T_FRIDGE",
    "WHITE_DENATURATION_K",
    "YOLK_DENATURATION_K",
    "YOLK_FULL_SET_K",
    "H_BOILING_SHELL",
    # geometry
    "create_circular_mesh",
    "create_egg_mesh",
    "egg_profile",
    "mesh_volume",
    # solver
    "HeatSolver",
    "solve_condense",
    "mass_axisym",
    "stiffness_axisym",
    "robin_lhs_axisym",
    "robin_rhs_axisym",
    # doneness
    "DONENESS_DEFINITIONS",
    "MEDIUM_YOLK_FRACTION",
    "center_crossing_times",
    "crossing_time",
    "doneness_times",
    # options (finance)
    "black_scholes_call",
    "solve_black_scholes_heat_fem",
    "fourier_sphere_center_temp",
    "ThermalResponse",
    "KitchenUncertainty",
    "check_similarity_against_fem",
    "price_egg_corridor_option",
    "egg_culinary_payoff",
    "solve_american_egg_option",
]
