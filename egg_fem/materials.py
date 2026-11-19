"""Temperature-dependent material properties for egg components.

All properties are given as functions of temperature T [K].

Provenance of the property fits.  The linear temperature fits below are
representative values chosen to match the range of published data for liquid
egg white and yolk; they are not coefficients copied from a single paper.
Sources whose data and ranges informed the choice:

  - Coimbra, J. S. R., Gabas, A. L., Minim, L. A., Garcia Rojas, E. E.,
    Telis, V. R. N., Telis-Romero, J. (2006).  Density, heat capacity and
    thermal conductivity of liquid egg products.  Journal of Food
    Engineering 74(2), 186-190.  doi:10.1016/j.jfoodeng.2005.01.043
    (measured between 273 and 311 K, i.e. 0-38 °C).
  - Denys, S., Pieters, J. G., Dewettinck, K. (2004).  Computational fluid
    dynamics analysis of combined conductive and convective heat transfer in
    model eggs.  Journal of Food Engineering 63(3), 281-290.
    doi:10.1016/j.jfoodeng.2003.06.002.

Caveats: above 38 °C the fits are extrapolations beyond the measured range,
and they describe raw (liquid) egg, yet they are used here after the egg has
coagulated.  The simulation inherits that uncertainty.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class EggMaterial:
    """Material properties for an egg component (white or yolk).

    Each property can be either a constant or a callable T -> value.
    Optionally includes endothermic latent heat of protein denaturation
    modeled via an apparent (effective) heat capacity peak around
    `denaturation_temp` with standard deviation `denaturation_width`.
    """

    name: str
    density: Callable  # ρ [kg/m³]
    conductivity: Callable  # κ [W/(m·K)]
    specific_heat: Callable  # Cp [J/(kg·K)]
    latent_heat: float = 0.0  # ΔH [J/kg] (endothermic enthalpy of denaturation)
    denaturation_temp: float = 336.15  # T_d [K]
    denaturation_width: float = 3.0  # σ [K] (Gaussian width of phase transition)

    def effective_specific_heat(self, T):
        """Effective specific heat Cp_eff(T) [J/(kg·K)] including latent heat."""
        cp = self.specific_heat(T)
        if self.latent_heat > 0.0:
            dT = T - self.denaturation_temp
            sigma = self.denaturation_width
            cp = cp + (self.latent_heat / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
                -0.5 * (dT / sigma) ** 2
            )
        return cp

    def rho_cp(self, T):
        """Product ρ·Cp_eff at temperature T."""
        return self.density(T) * self.effective_specific_heat(T)


# ---------------------------------------------------------------------------
# Reference temperatures [K] & Boundary Constants
# ---------------------------------------------------------------------------

T_BOILING = 373.15  # 100 °C
T_ROOM = 293.15  # 20 °C
T_FRIDGE = 281.15  # 8 °C

WHITE_DENATURATION_K = 336.15  # 63 °C – egg white sets
YOLK_DENATURATION_K = 338.15  # 65 °C – yolk begins to solidify
YOLK_FULL_SET_K = 343.15  # 70 °C – yolk fully hard

# Effective shell heat transfer coefficient [W/(m²·K)] combining boiling-water
# convection and eggshell conduction (~0.35 mm, κ ≈ 2.25 W/(m·K), i.e. about
# 6400 W/(m²·K) for the shell alone).  The 3000 W/(m²·K) water-side value is an
# assumption; published effective coefficients for eggs in hot water are about
# 1000-2100 W/(m²·K) (e.g. 2072 W/(m²·K) in "Numerical modeling of heat
# transfer and pasteurizing value during thermal processing of intact egg",
# PMC4708634, measured at 60 °C).  The Biot number h*R/k is large (about 35 to
# 126 for h = 1000 to 3000 W/(m²·K), R = 21 mm, k = 0.5 to 0.6 W/(m·K)), but
# the film still matters at the level of tens of seconds: relative to a
# Dirichlet (ideal-bath) shell the soft / medium / hard times of the default
# egg are later by 26 / 33 / 32 s at h = 1000, 13 / 17 / 16 s at h = 2000 and
# 9 / 11 / 11 s at h = 3000 W/(m²·K) (notebook 03).
H_BOILING_SHELL = 2000.0


# ---------------------------------------------------------------------------
# Temperature-dependent properties (linear fits from experimental data)
# ---------------------------------------------------------------------------

def _white_density(T):
    """Egg white density [kg/m³]. Decreases slightly with temperature."""
    return 1040.0 - 0.12 * (T - 293.15)


def _white_conductivity(T):
    """Egg white thermal conductivity [W/(m·K)]."""
    return 0.50 + 0.00125 * (T - 293.15)


def _white_specific_heat(T):
    """Egg white specific heat [J/(kg·K)]."""
    return 3800.0 + 1.2 * (T - 293.15)


def _yolk_density(T):
    """Egg yolk density [kg/m³]."""
    return 1030.0 - 0.125 * (T - 293.15)


def _yolk_conductivity(T):
    """Egg yolk thermal conductivity [W/(m·K)]."""
    return 0.32 + 0.0005 * (T - 293.15)


def _yolk_specific_heat(T):
    """Egg yolk specific heat [J/(kg·K)]."""
    return 3300.0 + 0.8 * (T - 293.15)


egg_white_properties = EggMaterial(
    name="egg_white",
    density=_white_density,
    conductivity=_white_conductivity,
    specific_heat=_white_specific_heat,
)

egg_yolk_properties = EggMaterial(
    name="egg_yolk",
    density=_yolk_density,
    conductivity=_yolk_conductivity,
    specific_heat=_yolk_specific_heat,
)

# Material presets incorporating endothermic protein denaturation enthalpy
egg_white_denaturation_properties = EggMaterial(
    name="egg_white_denaturation",
    density=_white_density,
    conductivity=_white_conductivity,
    specific_heat=_white_specific_heat,
    latent_heat=2000.0,  # ~2.0 kJ/kg (~18 J/g dry protein * ~11% protein)
    denaturation_temp=WHITE_DENATURATION_K,
    denaturation_width=3.0,
)

egg_yolk_denaturation_properties = EggMaterial(
    name="egg_yolk_denaturation",
    density=_yolk_density,
    conductivity=_yolk_conductivity,
    specific_heat=_yolk_specific_heat,
    latent_heat=1800.0,  # ~1.8 kJ/kg (~11 J/g dry protein * ~16% protein)
    denaturation_temp=YOLK_DENATURATION_K,
    denaturation_width=3.0,
)


# ---------------------------------------------------------------------------
# Constant (averaged) properties for simple models
# ---------------------------------------------------------------------------

WHITE_CONST = EggMaterial(
    name="egg_white_const",
    density=lambda T: 1038.0,
    conductivity=lambda T: 0.55,
    specific_heat=lambda T: 3800.0,
)

YOLK_CONST = EggMaterial(
    name="egg_yolk_const",
    density=lambda T: 1025.0,
    conductivity=lambda T: 0.34,
    specific_heat=lambda T: 3300.0,
)
