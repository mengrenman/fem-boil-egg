"""Arrhenius reaction kinetics for egg protein denaturation.

Models protein coagulation as a first-order irreversible rate process:
    dα/dt = k(T) · (1 - α),    k(T) = A · exp(-E_a / (R · T))

where α(x, t) ∈ [0, 1] is the local degree of denaturation (0 = native/raw,
1 = fully coagulated).  Along any nodal temperature history T(t), the exact
integrated conversion is:
    α(t) = 1 - exp(-∫_0^t k(T(τ)) dτ)
"""

from dataclasses import dataclass
import numpy as np
from skfem import Basis, ElementTriP1, Functional, asm

from .materials import WHITE_DENATURATION_K, YOLK_DENATURATION_K

R_GAS = 8.314462618  # Universal gas constant [J/(mol·K)]


@dataclass
class ArrheniusKinetics:
    """First-order Arrhenius kinetic parameters for protein denaturation.

    Parameters
    ----------
    name : str
        Component identifier (e.g. ``"egg_white"`` or ``"egg_yolk"``).
    E_a : float
        Activation energy [J/mol].
    A : float
        Pre-exponential frequency factor [1/s].
    """

    name: str
    E_a: float
    A: float

    @classmethod
    def from_reference_half_life(cls, name: str, E_a: float, T_ref: float, half_life: float):
        """Construct kinetics calibrated to a target half-life [s] at T_ref [K]."""
        k_ref = np.log(2.0) / half_life
        A = k_ref * np.exp(E_a / (R_GAS * T_ref))
        return cls(name=name, E_a=E_a, A=float(A))

    def rate_constant(self, T):
        """Reaction rate constant k(T) [1/s] at temperature T [K]."""
        T_arr = np.asarray(T, dtype=float)
        # Compute in log-space to avoid overflow in A * exp(...)
        log_k = np.log(self.A) - self.E_a / (R_GAS * T_arr)
        return np.exp(np.clip(log_k, -500.0, 100.0))

    def half_life(self, T):
        """Isothermal half-life τ_{1/2}(T) = ln(2) / k(T) [s]."""
        return np.log(2.0) / self.rate_constant(T)

    def isothermal_conversion(self, T, t):
        """Conversion α(t) = 1 - exp(-k(T)·t) at constant temperature T [K]."""
        k = self.rate_constant(T)
        return 1.0 - np.exp(-k * np.asarray(t, dtype=float))


# ---------------------------------------------------------------------------
# Calibrated kinetic presets for Egg White and Egg Yolk
# ---------------------------------------------------------------------------
# Egg white proteins (ovalbumin / ovotransferrin) have high activation energy
# (~430 kJ/mol), giving a sharp thermal transition above 64 °C.
# Egg yolk proteins (livetins / LDL) have lower activation energy (~250 kJ/mol),
# allowing gradual custard-like thickening below 64 °C (crossover at ~63.3 °C).

WHITE_KINETICS = ArrheniusKinetics.from_reference_half_life(
    name="egg_white",
    E_a=480_000.0,
    T_ref=WHITE_DENATURATION_K,  # 336.15 K (63 °C)
    half_life=900.0,   # 15 min at 63 °C -> ~52 s at 68 °C
)

YOLK_KINETICS = ArrheniusKinetics.from_reference_half_life(
    name="egg_yolk",
    E_a=250_000.0,
    T_ref=YOLK_DENATURATION_K,  # 338.15 K (65 °C)
    half_life=360.0,   # 6 min at 65 °C -> ~10.4 min at 63 °C (crossover at ~64.5 °C)
)


# ---------------------------------------------------------------------------
# Axisymmetric volume integration & spatiotemporal conversion history
# ---------------------------------------------------------------------------

@Functional
def _axisym_field_integral(w):
    r = np.maximum(w.x[0], 1e-12)
    return w["field"] * 2.0 * np.pi * r


@Functional
def _axisym_volume(w):
    r = np.maximum(w.x[0], 1e-12)
    return 2.0 * np.pi * r


def compute_denaturation_history(
    mesh,
    times,
    temperatures,
    white_kinetics: ArrheniusKinetics = WHITE_KINETICS,
    yolk_kinetics: ArrheniusKinetics = YOLK_KINETICS,
):
    """Integrate Arrhenius denaturation α(r, z, t) over a temperature history.

    Parameters
    ----------
    mesh : skfem.MeshTri
        Axisymmetric finite element mesh.
    times : sequence of float
        Simulation time stamps [s], shape ``(n_steps,)``.
    temperatures : sequence of ndarray
        Nodal temperature vectors [K] at each time step, shape ``(n_steps, n_nodes)``.
    white_kinetics, yolk_kinetics : ArrheniusKinetics
        Kinetic models for the white and yolk regions.

    Returns
    -------
    result : dict
        Dictionary containing:
        - ``"alpha"``: ndarray of shape ``(n_steps, n_nodes)``, nodal conversion ∈ [0, 1]
        - ``"alpha_white_avg"``: ndarray of shape ``(n_steps,)``, volume-averaged white conversion
        - ``"alpha_yolk_avg"``: ndarray of shape ``(n_steps,)``, volume-averaged yolk conversion
        - ``"alpha_center"``: ndarray of shape ``(n_steps,)``, conversion at yolk center
        - ``"perfection_index"``: ndarray of shape ``(n_steps,)``,
          ``alpha_white_avg * (1 - alpha_yolk_avg)`` measuring soft-boiled optimality
    """
    times_arr = np.asarray(times, dtype=float)
    T_hist = np.asarray(temperatures, dtype=float)  # (n_steps, n_nodes)
    n_steps, n_nodes = T_hist.shape
    dt_steps = np.diff(times_arr)[:, None]  # (n_steps - 1, 1)

    # Evaluate white and yolk conversion histories at all nodes independently
    rates_w = white_kinetics.rate_constant(T_hist)
    rates_y = yolk_kinetics.rate_constant(T_hist)

    cum_w = np.zeros_like(T_hist)
    cum_y = np.zeros_like(T_hist)
    if n_steps > 1:
        cum_w[1:] = np.cumsum(0.5 * (rates_w[:-1] + rates_w[1:]) * dt_steps, axis=0)
        cum_y[1:] = np.cumsum(0.5 * (rates_y[:-1] + rates_y[1:]) * dt_steps, axis=0)

    alpha_w_nodes = 1.0 - np.exp(-cum_w)
    alpha_y_nodes = 1.0 - np.exp(-cum_y)

    # Combined nodal field for plotting (yolk nodes get alpha_y, white nodes get alpha_w)
    alpha = alpha_w_nodes.copy()
    if mesh.subdomains and "yolk" in mesh.subdomains:
        yolk_nodes = np.unique(mesh.t[:, mesh.subdomains["yolk"]])
        alpha[:, yolk_nodes] = alpha_y_nodes[:, yolk_nodes]

    # Compute axisymmetric volume-weighted averages over white and yolk subdomains
    elem = ElementTriP1()
    basis = Basis(mesh, elem)

    if mesh.subdomains and "white" in mesh.subdomains and "yolk" in mesh.subdomains:
        basis_white = Basis(mesh, elem, elements=mesh.subdomains["white"])
        basis_yolk = Basis(mesh, elem, elements=mesh.subdomains["yolk"])
        vol_white = float(asm(_axisym_volume, basis_white))
        vol_yolk = float(asm(_axisym_volume, basis_yolk))

        alpha_white_avg = np.empty(n_steps)
        alpha_yolk_avg = np.empty(n_steps)
        for i in range(n_steps):
            aw_field = basis_white.interpolate(alpha_w_nodes[i])
            ay_field = basis_yolk.interpolate(alpha_y_nodes[i])
            alpha_white_avg[i] = float(asm(_axisym_field_integral, basis_white, field=aw_field)) / vol_white
            alpha_yolk_avg[i] = float(asm(_axisym_field_integral, basis_yolk, field=ay_field)) / vol_yolk
    else:
        vol_total = float(asm(_axisym_volume, basis))
        alpha_white_avg = np.empty(n_steps)
        for i in range(n_steps):
            a_field = basis.interpolate(alpha_w_nodes[i])
            alpha_white_avg[i] = float(asm(_axisym_field_integral, basis, field=a_field)) / vol_total
        alpha_yolk_avg = alpha_white_avg.copy()

    # Center node conversion
    coords = mesh.p
    center_dof = int(np.argmin(coords[0] ** 2 + coords[1] ** 2))
    alpha_center = alpha[:, center_dof]

    perfection_index = alpha_white_avg * (1.0 - alpha_yolk_avg)

    return {
        "alpha": alpha,
        "alpha_white_avg": alpha_white_avg,
        "alpha_yolk_avg": alpha_yolk_avg,
        "alpha_center": alpha_center,
        "perfection_index": perfection_index,
    }
