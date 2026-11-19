"""Options-style problems built on the egg heat equation.

Bridges the axisymmetric FEM heat equation with options pricing ideas:

  1. Black-Scholes <-> heat equation duality (1D FEM solver vs closed form).
  2. A binary corridor ("soft-boil warranty") priced under kitchen uncertainty.
  3. An optimal stopping problem ("when to pull the egg") solved with
     Longstaff-Schwartz least-squares Monte Carlo.

What is modeled
---------------
The egg is not a traded asset and none of this is market-calibrated.  The
finance vocabulary is borrowed for the payoff structures; the numbers are
expected culinary utilities, not prices.

The Monte Carlo layers (2 and 3) are driven by the FEM, not by a closed-form
surrogate.  One reference FEM run of the default egg (20 °C egg, 100 °C
Dirichlet bath) is condensed into a dimensionless response

    theta(t) = (T_probe(t) - T_bath) / (T_init - T_bath),

stored in a :class:`ThermalResponse`.  Other kitchens (a different starting
temperature, bath temperature or egg size) are obtained by similarity
scaling

    T(t; T_init, T_bath, s) = T_bath + (T_init - T_bath) * theta(t / s**2),

with ``s`` the linear size scale.  For constant material properties this is
exact; because the properties depend on temperature it is an approximation,
checked against direct FEM runs by :func:`check_similarity_against_fem`.  The
error is systematic, not random: for the default egg, cold eggs and cool baths
come out early and warm eggs late by up to a few seconds, which shrinks the
spread between kitchens (notebook 04 quantifies the effect on the prices).

Kitchen uncertainty (starting temperature, bath temperature, egg size) is
described by one :class:`KitchenUncertainty` model shared by the corridor and
the stopping problem.  Everything is undiscounted unless a discount rate is
passed explicitly.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.sparse.linalg import spsolve
from skfem import (
    Basis,
    BilinearForm,
    ElementLineP1,
    MeshLine,
    asm,
)
from skfem.helpers import dot, grad

from .doneness import crossing_time
from .geometry import create_egg_mesh
from .materials import (
    T_BOILING,
    T_ROOM,
    egg_white_properties,
    egg_yolk_properties,
)
from .solver import HeatSolver

_KELVIN = 273.15


# ---------------------------------------------------------------------------
# 1. Black-Scholes <-> 1D Heat Equation FEM Duality
# ---------------------------------------------------------------------------

def black_scholes_call(S, K: float, T: float, r: float, sigma: float):
    """Exact Black-Scholes formula for a European call option.

    Parameters
    ----------
    S : float or array_like
        Spot price(s); must be non-negative (``S = 0`` is worth 0).
    K : float
        Strike; must be positive.
    T : float
        Time to expiry in years.  ``T = 0`` returns the intrinsic value
        ``max(S - K, 0)``; negative ``T`` is an error.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility; must be positive.

    Returns
    -------
    float or ndarray
        Call value(s), same shape as ``S``.

    Raises
    ------
    ValueError
        For ``sigma <= 0``, ``K <= 0``, ``T < 0``, non-finite ``r``, or
        negative ``S``.
    """
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}")
    if not (np.isfinite(K) and K > 0.0):
        raise ValueError(f"K must be positive and finite, got {K!r}")
    if not (np.isfinite(T) and T >= 0.0):
        raise ValueError(f"T must be finite and >= 0, got {T!r}")
    if not np.isfinite(r):
        raise ValueError(f"r must be finite, got {r!r}")
    S_arr = np.asarray(S, dtype=float)
    if np.any(S_arr < 0.0) or not np.all(np.isfinite(S_arr)):
        raise ValueError("S must be finite and non-negative")
    if T == 0.0:
        return np.maximum(S_arr - K, 0.0)
    positive = S_arr > 0.0
    S_safe = np.where(positive, S_arr, 1.0)
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S_safe * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(positive, price, 0.0)


@BilinearForm
def _mass_1d(u, v, w):
    return u * v


@BilinearForm
def _stiff_1d(u, v, w):
    return dot(grad(u), grad(v))


# Largest allowed magnitude of the exponent that scales the transformed
# solution (see solve_black_scholes_heat_fem).  The transformed solution
# u(x, tau) spans a factor exp(E) between the two ends of the domain, so the
# double-precision round-off in the linear solve grows with exp(E).  Measured
# max error against the closed form (K = 100, 801 nodes, 400 steps, domain
# x in [-2.5, 2.5]): about 1e-3 USD at E = 21, 6e-3 at E = 36, 0.03 at E = 51,
# 0.12 at E = 79, 7.8 at E = 314 and NaN (overflow) at E = 1251.
_BS_MAX_EXPONENT = 40.0


def solve_black_scholes_heat_fem(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.25,
    x_min: float = -2.5,
    x_max: float = 2.5,
    n_nodes: int = 401,
    n_steps: int = 200,
):
    """Solve the Black-Scholes PDE by transforming it into the 1D heat equation
    and marching forward in τ with scikit-fem P1 finite elements.

    Coordinate transform:
        x = ln(S / K)
        τ = ½ σ² (T - t) ∈ [0, ½ σ² T]
        k = 2 r / σ²
        α = -½ (k - 1),   β = -¼ (k + 1)²
        V(S, t) = K · exp(α x + β τ) · u(x, τ)

    Then u(x, τ) satisfies the canonical heat equation:
        ∂u/∂τ = ∂²u/∂x²
    with initial condition u(x, 0) = max(exp(½(k+1)x) - exp(½(k-1)x), 0).

    Accuracy.  Space is discretized with P1 elements (second order in the
    node spacing) and time with backward Euler, which is only first order in
    the time step: halving ``n_steps`` doubles the time-discretization error.
    For the default call with 401 nodes the max error against the closed
    form on S in [50, 150] is 0.0059 USD at 200 steps and 0.0035 USD at 400
    steps.  These are errors at the mesh nodes (the error at a point between
    two nodes is larger, about 0.0057 USD at 400 steps), and they are mostly
    time-discretization error.

    Numerical range.  The transformed solution varies over a factor of about
    exp(E), with ``E = ½ (|k| + 1) max(|x_min|, x_max)`` plus a small τ term.
    For small σ (large ``k``) this factor is astronomically large: the
    round-off error grows with it (about 0.03 USD at E = 51 and 8 USD at
    E = 314 with 801 nodes) and the exponentials overflow near E = 700.
    Instead of returning inaccurate or non-finite values, the function raises
    ``ValueError`` when E exceeds 40.  A narrower domain (smaller ``x_max``)
    lowers E in proportion; it must still cover several multiples of
    ``σ √T`` on each side of ``x = 0``.  The guard only protects against
    round-off and overflow: a large ``k`` also needs more nodes to be
    resolved (with 401 nodes the max error on S in [50, 150] is 0.051 USD at
    200 steps and 0.065 USD at 400 steps for σ = 0.06, and 1.2 USD at 200 steps
    and 0.48 USD at 400 steps for σ = 0.25 with r = 0.5), so compare with
    ``V_exact``.

    Parameters
    ----------
    K, T, r, sigma : float
        Strike, expiry [years], risk-free rate and volatility.
        ``K``, ``T`` and ``sigma`` must be positive.
    x_min, x_max : float
        Ends of the log-moneyness domain; ``x_min < 0 < x_max``.
    n_nodes : int
        Number of mesh nodes, at least 3.
    n_steps : int
        Number of backward-Euler steps, at least 1.

    Returns
    -------
    result : dict
        - ``"S"``: stock price grid
        - ``"V_fem"``: FEM option value at t = 0
        - ``"V_exact"``: closed-form Black-Scholes call price at t = 0
        - ``"x"``: log-moneyness grid x = ln(S/K)
        - ``"u_init"``: transformed heat initial condition u(x, 0)
        - ``"u_final"``: diffused heat solution u(x, τ_max)

    Raises
    ------
    ValueError
        For invalid inputs, or when the transformed problem is too badly
        scaled for the domain (``k`` too large; see above).
    """
    for name, value in (("K", K), ("T", T), ("sigma", sigma)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if not np.isfinite(r):
        raise ValueError(f"r must be finite, got {r!r}")
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_min < 0.0 < x_max):
        raise ValueError(
            f"need x_min < 0 < x_max (finite), got x_min={x_min!r}, x_max={x_max!r}"
        )
    if int(n_nodes) != n_nodes or n_nodes < 3:
        raise ValueError(f"n_nodes must be an integer >= 3, got {n_nodes!r}")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    n_nodes = int(n_nodes)
    n_steps = int(n_steps)

    k = 2.0 * r / (sigma**2)
    alpha = -0.5 * (k - 1.0)
    beta = -0.25 * (k + 1.0) ** 2
    tau_max = 0.5 * (sigma**2) * T
    dtau = tau_max / n_steps

    exponent = 0.5 * (abs(k) + 1.0) * max(-x_min, x_max) - beta * tau_max
    if not (np.isfinite(exponent) and exponent <= _BS_MAX_EXPONENT):
        raise ValueError(
            f"k = 2r/sigma^2 = {k:.4g} is too large for the domain "
            f"[{x_min}, {x_max}]: the log-space transform scales the solution "
            f"by exp({exponent:.4g}) (limit exp({_BS_MAX_EXPONENT:g})), which "
            "overflows or destroys the accuracy.  Use a larger sigma, a "
            "narrower x-domain, or the closed-form black_scholes_call."
        )

    x_grid = np.linspace(x_min, x_max, n_nodes)
    mesh = MeshLine(x_grid)
    basis = Basis(mesh, ElementLineP1())

    M = asm(_mass_1d, basis)
    A_stiff = asm(_stiff_1d, basis)
    lhs = M / dtau + A_stiff

    # Initial condition at tau = 0 (option payoff at maturity t = T)
    u = np.maximum(
        np.exp(0.5 * (k + 1.0) * x_grid) - np.exp(0.5 * (k - 1.0) * x_grid),
        0.0,
    )
    u_init = u.copy()

    bd_dofs = np.array([0, n_nodes - 1], dtype=int)
    interior = np.arange(1, n_nodes - 1, dtype=int)
    lhs_int = lhs[interior][:, interior].tocsc()

    for step in range(1, n_steps + 1):
        tau = step * dtau
        # Asymptotic Dirichlet boundary conditions at x_min and x_max
        S_max = K * np.exp(x_max)
        V_max = S_max - K * np.exp(-r * (2.0 * tau / sigma**2))
        u_bc = np.zeros_like(u)
        u_bc[0] = 0.0
        u_bc[-1] = (V_max / K) * np.exp(-alpha * x_max - beta * tau)

        rhs = M @ u / dtau
        rhs_int = rhs[interior] - lhs[interior][:, bd_dofs] @ u_bc[bd_dofs]
        u = u_bc
        u[interior] = spsolve(lhs_int, rhs_int)

    S_grid = K * np.exp(x_grid)
    V_fem = K * np.exp(alpha * x_grid + beta * tau_max) * u
    if not np.all(np.isfinite(V_fem)):
        raise ValueError(
            f"the heat-equation solution is not finite (k = {k:.4g} is too "
            "large for this domain); use a larger sigma or a narrower domain."
        )
    V_exact = black_scholes_call(S_grid, K=K, T=T, r=r, sigma=sigma)

    return {
        "S": S_grid,
        "V_fem": V_fem,
        "V_exact": V_exact,
        "x": x_grid,
        "u_init": u_init,
        "u_final": u,
    }


# ---------------------------------------------------------------------------
# 2a. Exact center temperature of a homogeneous sphere
# ---------------------------------------------------------------------------

# Fourier number alpha*t/R^2 at which fourier_sphere_center_temp switches from
# the small-time (image) form to the Fourier series.  At 0.1 the series needs
# 8 terms and the image form 3 terms for machine precision.
_FO_SWITCH = 0.1
_FO_MIN = 1e-12  # below this the center is still at T_init to machine precision
_SPHERE_DEFAULT_TERMS = 12


def fourier_sphere_center_temp(
    t,
    T_init: float = 293.15,
    T_bath: float = 373.15,
    R_eff: float = 0.0228,
    alpha_diff: float = 1.32e-7,
    n_terms=None,
):
    """Exact center temperature [K] of a homogeneous sphere in a hot bath.

    A sphere of radius ``R_eff`` and constant thermal diffusivity
    ``alpha_diff`` starts at the uniform temperature ``T_init`` and has its
    surface held at ``T_bath`` from t = 0.  With the Fourier number
    ``Fo = alpha_diff * t / R_eff**2``, the center temperature is

        T = T_bath + (T_init - T_bath) * theta(Fo),

    where theta is given by two equivalent exact expressions:

    * for ``Fo >= 0.1`` the Fourier series
      ``theta = 2 * sum_{n>=1} (-1)**(n+1) * exp(-n**2 * pi**2 * Fo)``;
    * for ``Fo < 0.1`` the image (small-time) form
      ``theta = 1 - 2 / sqrt(pi * Fo) * sum_{m>=0} exp(-(2m+1)**2 / (4 Fo))``,
      because the Fourier series converges too slowly there (and is not
      convergent at t = 0).

    The two agree to machine precision at the switch point.  The result is
    correct for every ``t >= 0`` regardless of ``n_terms`` and equals
    ``T_init`` exactly at t = 0.  This is the analytic benchmark for the FEM
    on a homogeneous sphere (notebook 01) and the analytic response in
    :meth:`ThermalResponse.from_sphere`.  It is *not* calibrated to the egg:
    a sphere has no yolk, no egg shape and no temperature-dependent
    properties.  The default radius and diffusivity describe a generic
    22.8 mm sphere.

    Parameters
    ----------
    t : float or array_like
        Time [s]; negative values are treated as 0.
    T_init, T_bath : float or array_like
        Initial and surface temperatures [K].
    R_eff : float or array_like
        Sphere radius [m]; must be positive.
    alpha_diff : float or array_like
        Thermal diffusivity [m²/s]; must be positive.
    n_terms : int or None
        Number of Fourier terms used where ``Fo >= 0.1``.  ``None`` (default)
        uses 12, whose truncation error there is far below machine
        precision; a smaller number gives a truncated series at the
        corresponding accuracy.  The small-time form always uses
        enough terms.  All of ``t``, ``T_init``, ``T_bath``, ``R_eff`` and
        ``alpha_diff`` broadcast against each other.

    Returns
    -------
    float or ndarray
        Center temperature [K], within ``[min(T_init, T_bath), max(...)]``.
    """
    if n_terms is None:
        n_terms = _SPHERE_DEFAULT_TERMS
    if int(n_terms) != n_terms or n_terms < 1:
        raise ValueError(f"n_terms must be a positive integer or None, got {n_terms!r}")
    n_terms = int(n_terms)

    t_arr, T_init_arr, T_bath_arr, R_arr, alpha_arr = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (t, T_init, T_bath, R_eff, alpha_diff))
    )
    if np.any(~(R_arr > 0.0)) or np.any(~(alpha_arr > 0.0)):
        raise ValueError("R_eff and alpha_diff must be positive")

    fo = np.maximum(t_arr, 0.0) * alpha_arr / R_arr**2

    # Fourier series (large Fo).
    fo_late = np.where(fo >= _FO_SWITCH, fo, 1.0)
    theta_late = np.zeros_like(fo)
    for n in range(1, n_terms + 1):
        sign = 1.0 if n % 2 == 1 else -1.0
        theta_late = theta_late + sign * np.exp(-((n * np.pi) ** 2) * fo_late)
    theta_late = 2.0 * theta_late

    # Image form (small Fo).  Three terms are enough for Fo < 0.1: the
    # fourth is below exp(-120).
    fo_early = np.maximum(np.where(fo < _FO_SWITCH, fo, _FO_SWITCH), _FO_MIN)
    images = np.zeros_like(fo)
    for m in range(4):
        images = images + np.exp(-((2 * m + 1) ** 2) / (4.0 * fo_early))
    theta_early = 1.0 - 2.0 / np.sqrt(np.pi * fo_early) * images

    theta = np.clip(np.where(fo >= _FO_SWITCH, theta_late, theta_early), 0.0, 1.0)
    # theta = 1 gives exactly T_init (no round-off from T_bath + (T_init - T_bath)).
    T_center = theta * T_init_arr + (1.0 - theta) * T_bath_arr
    return T_center[()] if T_center.ndim == 0 else T_center


# ---------------------------------------------------------------------------
# 2b. FEM-derived thermal response, kitchen uncertainty
# ---------------------------------------------------------------------------

_PROBES = ("center", "white_min", "yolk_min")
# Probes that are the minimum over a region.  T = T_bath + (T_init - T_bath) *
# theta(x, t) has its minimum where theta is largest only when T_init < T_bath
# (heating); for a cooling egg the minimum moves to another point, so the
# similarity scaling of such a probe is not valid.
_EXTREMAL_PROBES = ("white_min", "yolk_min")


def _default_materials():
    return {"white": egg_white_properties, "yolk": egg_yolk_properties}


def _fem_probe_history(solver, probe):
    """Return (times [s], probe temperature [°C]) from a finished HeatSolver."""
    if probe == "center":
        return solver.center_temperature_history()
    if probe in ("white_min", "yolk_min"):
        return solver.region_min_temperature_history(probe.split("_")[0])
    raise ValueError(f"probe must be one of {_PROBES}, got {probe!r}")


class ThermalResponse:
    """Dimensionless heating curve ``theta(t)`` of one probe point or region.

    ``theta(t) = (T_probe(t) - T_bath) / (T_init - T_bath)`` is 1 at t = 0 and
    decays to 0.  It is stored on a time grid and evaluated by linear
    interpolation.  Calling the object applies the similarity scaling

        T(t; T_init, T_bath, s) = T_bath + (T_init - T_bath) * theta(t / s**2)

    to give the probe temperature for a kitchen with a different starting
    temperature, bath temperature and linear size scale ``s``.  For constant
    material properties this is exact; with temperature-dependent properties
    it is an approximation (see :func:`check_similarity_against_fem`).

    Build one with :meth:`from_fem` (the FEM response of an egg mesh, the
    intended use) or :meth:`from_sphere` (the exact homogeneous-sphere
    response, fast, useful for tests).

    Parameters
    ----------
    times_s : array_like
        Strictly increasing times [s], starting at 0.
    theta : array_like
        Dimensionless response at those times; values must lie in [0, 1] and
        be non-increasing (the heating curve of a body in a hotter bath).
    probe : str
        Label of what ``theta`` describes (e.g. "white_min").
    """

    def __init__(self, times_s, theta, probe="custom"):
        times = np.array(times_s, dtype=float)
        theta = np.array(theta, dtype=float)
        if times.ndim != 1 or times.shape != theta.shape or len(times) < 2:
            raise ValueError("times_s and theta must be 1-D arrays of equal length >= 2")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(theta))):
            raise ValueError("times_s and theta must be finite")
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must start at 0 and be strictly increasing")
        if np.any(theta < -1e-12) or np.any(theta > 1.0 + 1e-12):
            raise ValueError("theta must lie in [0, 1]")
        if np.any(np.diff(theta) > 1e-9):
            raise ValueError("theta must be non-increasing")
        self.times_s = times
        self.theta = np.clip(theta, 0.0, 1.0)
        self.probe = str(probe)

    @property
    def t_end_s(self):
        """Last time on the response grid [s]."""
        return float(self.times_s[-1])

    def __repr__(self):
        return (
            f"ThermalResponse(probe={self.probe!r}, t_end_s={self.t_end_s:g}, "
            f"n_times={len(self.times_s)})"
        )

    @classmethod
    def from_fem(
        cls,
        probe="white_min",
        mesh=None,
        materials=None,
        T_init=T_ROOM,
        T_bath=T_BOILING,
        dt=1.0,
        t_end=1200.0,
    ):
        """Response of an egg mesh, from one FEM run with a Dirichlet bath.

        Parameters
        ----------
        probe : {"center", "white_min", "yolk_min"}
            What is observed.  ``"center"`` is the temperature at the yolk
            center.  ``"white_min"`` is the coldest temperature in the white;
            for the default egg that coldest point is on the yolk surface (on
            the symmetry axis at the top of the yolk), so it is the temperature
            of the coolest point of the yolk surface, not of the yolk surface
            as a whole (the white is set when it reaches 63 °C, and at that
            moment even the coolest point of the yolk surface is at 63 °C,
            the soft-boil criterion of :mod:`egg_fem.doneness`).  The
            extremal probes ``"white_min"`` and ``"yolk_min"`` need a heated
            egg (``T_init < T_bath``).
            ``"yolk_min"`` is the coldest temperature in the yolk (its center
            for the default egg).
        mesh : skfem.MeshTri, optional
            Default ``create_egg_mesh()`` (55 x 42 mm egg, 28% yolk).
            ``"white_min"`` and ``"yolk_min"`` need "white" and "yolk"
            subdomains.
        materials : dict, optional
            Default ``{"white": egg_white_properties, "yolk": egg_yolk_properties}``.
        T_init, T_bath : float
            Reference starting temperature and bath temperature [K]
            (default 20 °C and 100 °C).
        dt : float
            Time step [s] of the backward-Euler run.
        t_end : float
            Length of the run [s].  Scaled kitchens are evaluated at
            ``t / s**2``, so ``t_end`` must exceed the largest such time.

        Returns
        -------
        ThermalResponse
            The FEM start can undershoot ``T_init`` slightly (consistent mass
            matrix, see :class:`HeatSolver`): by 0.03 K on the default mesh
            and up to 0.02 K at the center of a 3 mm mesh.  ``theta`` is
            therefore clipped to [0, 1] and made non-increasing, which
            changes it by less than 5e-4 on these meshes.
        """
        if probe not in _PROBES:
            raise ValueError(f"probe must be one of {_PROBES}, got {probe!r}")
        if not (np.isfinite(T_init) and np.isfinite(T_bath)) or T_init == T_bath:
            raise ValueError("T_init and T_bath must be finite and different")
        if probe in _EXTREMAL_PROBES and T_init > T_bath:
            raise ValueError(
                f"probe {probe!r} is a regional minimum and needs a heated egg "
                "(T_init < T_bath)"
            )
        if not (np.isfinite(t_end) and t_end > 0.0):
            raise ValueError("t_end must be positive")
        if mesh is None:
            mesh = create_egg_mesh()
        if materials is None:
            materials = _default_materials()
        solver = HeatSolver(mesh, materials, T_init=T_init, T_boundary=T_bath, dt=dt)
        solver.run(t_end, progress=False)
        times, T_probe_C = _fem_probe_history(solver, probe)
        theta = (T_probe_C + _KELVIN - T_bath) / (T_init - T_bath)
        theta = np.minimum.accumulate(np.clip(theta, 0.0, 1.0))
        return cls(times, theta, probe=probe)

    @classmethod
    def from_sphere(cls, R_eff, alpha_diff, t_end, n_times):
        """Exact response of the center of a homogeneous sphere.

        ``theta`` is evaluated with :func:`fourier_sphere_center_temp` on
        ``n_times`` equally spaced times in ``[0, t_end]``.  A sphere of
        radius ``R_eff * s`` has the response ``theta(t / s**2)``, so
        the similarity scaling is exact here.
        """
        if not (t_end > 0.0 and np.isfinite(t_end)):
            raise ValueError("t_end must be positive")
        if int(n_times) != n_times or n_times < 2:
            raise ValueError("n_times must be an integer >= 2")
        times = np.linspace(0.0, float(t_end), int(n_times))
        theta = fourier_sphere_center_temp(
            times, T_init=1.0, T_bath=0.0, R_eff=R_eff, alpha_diff=alpha_diff
        )
        return cls(times, theta, probe="sphere_center")

    def _check_heating(self, T0, Tb):
        # A regional-minimum probe is only defined for a heated egg (see _EXTREMAL_PROBES).
        if self.probe in _EXTREMAL_PROBES and np.any(T0 >= Tb):
            raise ValueError(
                f"probe {self.probe!r} is a regional minimum: the similarity scaling "
                "needs a heated egg (T_init < T_bath)"
            )

    def __call__(self, t_s, T_init_K, T_bath_K, size_scale=1.0):
        """Probe temperature [K] for a kitchen, by similarity scaling.

        All four arguments broadcast against each other (e.g. ``t_s`` of
        shape ``(1, m)`` with the kitchen parameters of shape ``(n, 1)``
        gives an ``(n, m)`` array).

        Raises
        ------
        ValueError
            If ``size_scale`` is not positive, a time is negative,
            ``t / size_scale**2`` lies beyond the end of the response grid
            (rebuild the response with a larger ``t_end``), or the probe is a
            regional minimum (``"white_min"``, ``"yolk_min"``) and a kitchen
            has ``T_init >= T_bath``.
        """
        t, T0, Tb, s = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (t_s, T_init_K, T_bath_K, size_scale))
        )
        if np.any(~(s > 0.0)):
            raise ValueError("size_scale must be positive")
        self._check_heating(T0, Tb)
        if np.any(t < 0.0):
            raise ValueError("times must be non-negative")
        t_ref = t / s**2
        if np.any(t_ref > self.t_end_s * (1.0 + 1e-12)):
            raise ValueError(
                f"t / size_scale**2 reaches {np.max(t_ref):.1f} s, beyond the "
                f"response grid (t_end = {self.t_end_s:.1f} s); build the response "
                "with a larger t_end"
            )
        theta = np.interp(t_ref, self.times_s, self.theta)
        T = Tb + (T0 - Tb) * theta
        return T[()] if T.ndim == 0 else T

    def time_to_reach(self, T_target_C, T_init_K, T_bath_K, size_scale=1.0):
        """Time [s] at which the probe first reaches ``T_target_C``.

        Returns 0 if the target is at or beyond the starting temperature,
        and ``nan`` if the probe does not reach it within the response grid.
        Arguments broadcast.
        """
        T_target, T0, Tb, s = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (T_target_C, T_init_K, T_bath_K, size_scale))
        )
        if np.any(~(s > 0.0)):
            raise ValueError("size_scale must be positive")
        self._check_heating(T0, Tb)
        theta_target = (T_target + _KELVIN - Tb) / (T0 - Tb)
        # theta is non-increasing in t, so reverse it for np.interp.
        t_ref = np.interp(theta_target, self.theta[::-1], self.times_s[::-1])
        t_ref = np.where(theta_target < self.theta[-1], np.nan, t_ref)
        t = t_ref * s**2
        return t[()] if t.ndim == 0 else t


@dataclass(frozen=True)
class KitchenUncertainty:
    """Kitchen-to-kitchen variation of the cooking conditions.

    One model shared by the corridor option and the stopping problem.

    Attributes
    ----------
    T_init_mean_C, T_init_std_C : float
        The egg's starting temperature is ``N(T_init_mean_C, T_init_std_C**2)``
        (default 20 °C, 3.5 °C: a cold-pantry to warm-kitchen spread; a
        refrigerator start at 8 °C is 3.4 standard deviations below the mean,
        so fridge eggs are outside the default model).
    T_bath_max_C : float
        Water temperature never exceeds this (default 100 °C, boiling at sea level).
    T_bath_shortfall_std_C : float
        The bath temperature is ``T_bath_max_C - |N(0, std**2)|`` (default
        std 1.2 °C): water that is below the boil by a half-normal amount,
        with no probability mass piled up at exactly 100 °C.
    size_cv : float
        The linear size scale ``s`` (``s = 1`` is the reference egg) is
        ``N(1, size_cv**2)``.  The default 0.02 corresponds to a mass
        coefficient of variation of about 6% (mass scales as ``s**3``).  This
        is a modeling choice, not a measured quantity: a uniform spread within
        one carton grade (US Large 56.7 to 63.8 g, EU M 53 to 63 g) would be
        a mass CV of 3.4% to 5.0%, i.e. ``size_cv`` of about 1.1% to 1.7%,
        while a mix of grades is much wider.  The corridor price depends
        strongly on it (notebook 04).  Must be below 0.15.
    """

    T_init_mean_C: float = 20.0
    T_init_std_C: float = 3.5
    T_bath_max_C: float = 100.0
    T_bath_shortfall_std_C: float = 1.2
    size_cv: float = 0.02

    def __post_init__(self):
        for name in ("T_init_mean_C", "T_init_std_C", "T_bath_max_C",
                     "T_bath_shortfall_std_C", "size_cv"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.T_init_std_C < 0.0 or self.T_bath_shortfall_std_C < 0.0:
            raise ValueError("standard deviations must be non-negative")
        if not (0.0 <= self.size_cv < 0.15):
            raise ValueError("size_cv must be in [0, 0.15)")

    def sample(self, n, rng=None):
        """Draw ``n`` kitchens.

        Parameters
        ----------
        n : int
            Number of kitchens.
        rng : numpy.random.Generator, int or None
            Random generator (or a seed).

        Returns
        -------
        T_init_K, T_bath_K, size_scale : ndarray
            Three arrays of shape ``(n,)``.
        """
        rng = np.random.default_rng(rng)
        T_init_K = _KELVIN + rng.normal(self.T_init_mean_C, self.T_init_std_C, size=n)
        shortfall = np.abs(rng.normal(0.0, self.T_bath_shortfall_std_C, size=n))
        T_bath_K = _KELVIN + self.T_bath_max_C - shortfall
        size_scale = rng.normal(1.0, self.size_cv, size=n)
        return T_init_K, T_bath_K, size_scale


def check_similarity_against_fem(
    response,
    T_init_K,
    T_bath_K,
    size_scale=1.0,
    thresholds_C=(63.0, 66.5, 70.0),
    geometry=None,
    materials=None,
    dt=1.0,
):
    """Compare the similarity-scaled response with a direct FEM run.

    A direct FEM run is made for the egg with every length scaled by
    ``size_scale`` (including the mesh sizes) and a time step ``dt *
    size_scale**2``, so both runs are the same discrete problem up to the
    temperature dependence of the material properties.  The result shows how
    much error the similarity scaling of :class:`ThermalResponse` makes.

    Parameters
    ----------
    response : ThermalResponse
        A response built by :meth:`ThermalResponse.from_fem` from the egg
        ``create_egg_mesh(**geometry)`` with time step ``dt``.
    T_init_K, T_bath_K, size_scale : float
        The perturbed kitchen.
    thresholds_C : sequence of float
        Temperatures whose crossing times are compared.
    geometry : dict, optional
        Keyword arguments of :func:`create_egg_mesh` for the reference egg
        (default: the defaults); lengths and mesh sizes are multiplied by
        ``size_scale``.
    materials : dict, optional
        Default white and yolk properties.
    dt : float
        Reference time step [s] used to build ``response``.

    Returns
    -------
    dict
        ``"times_s"`` (direct-run times), ``"T_direct_C"``, ``"T_scaled_C"``,
        ``"max_abs_diff_C"`` (over the common time range),
        ``"thresholds_C"``, ``"t_direct_s"``, ``"t_scaled_s"`` (crossing
        times) and ``"t_diff_s"`` (scaled minus direct).
    """
    probe = response.probe
    if probe not in _PROBES:
        raise ValueError(
            "the response must come from ThermalResponse.from_fem "
            f"(probe in {_PROBES}), got probe {probe!r}"
        )
    s = float(size_scale)
    if not s > 0.0:
        raise ValueError("size_scale must be positive")
    base = dict(
        length=0.055, breadth=0.042, yolk_radius=0.015, yolk_center_z=-0.002,
        mesh_size=0.0015, mesh_size_boundary=0.0006,
    )
    base.update(geometry or {})
    scaled = {}
    for key, value in base.items():
        scaled[key] = value * s if key != "asymmetry" else value
    mesh = create_egg_mesh(**scaled)
    solver = HeatSolver(
        mesh, materials or _default_materials(),
        T_init=T_init_K, T_boundary=T_bath_K, dt=dt * s**2,
    )
    solver.run(response.t_end_s * s**2, progress=False)
    times, T_direct_C = _fem_probe_history(solver, probe)
    keep = times <= response.t_end_s * s**2 * (1.0 + 1e-9)
    times, T_direct_C = times[keep], T_direct_C[keep]
    T_scaled_C = response(times, T_init_K, T_bath_K, s) - _KELVIN

    thresholds = [float(x) for x in thresholds_C]
    t_direct = np.array([crossing_time(times, T_direct_C, x) for x in thresholds])
    t_scaled = np.array([crossing_time(times, T_scaled_C, x) for x in thresholds])
    return {
        "times_s": times,
        "T_direct_C": T_direct_C,
        "T_scaled_C": T_scaled_C,
        "max_abs_diff_C": float(np.max(np.abs(T_scaled_C - T_direct_C))),
        "thresholds_C": np.array(thresholds),
        "t_direct_s": t_direct,
        "t_scaled_s": t_scaled,
        "t_diff_s": t_scaled - t_direct,
    }


def _check_response(response):
    if not callable(response):
        raise TypeError("response must be a ThermalResponse (or a callable with its signature)")


def _check_path_count(name, n):
    if int(n) != n or n < 2:
        raise ValueError(f"{name} must be an integer >= 2, got {n!r}")
    return int(n)


def _simulate_probe_C(response, t_s, kitchens):
    """True probe temperatures [°C], shape (n_paths, len(t_s))."""
    T_init_K, T_bath_K, size_scale = kitchens
    return (
        response(t_s[None, :], T_init_K[:, None], T_bath_K[:, None], size_scale[:, None])
        - _KELVIN
    )


# ---------------------------------------------------------------------------
# 2c. Corridor ("soft-boil warranty") option
# ---------------------------------------------------------------------------

def price_egg_corridor_option(
    cook_times_s,
    response,
    K_low_C: float = 63.0,
    K_high_C: float = 70.0,
    notional: float = 10.0,
    uncertainty=None,
    n_paths: int = 20000,
    seed: int = 42,
):
    """Price a binary corridor ("soft-boil warranty") over pull times.

    Payoff for pulling the egg at time ``t``:

        notional   if  K_low_C <= T_probe(t) <= K_high_C
        0          otherwise

    The default probe of the intended :meth:`ThermalResponse.from_fem`
    response is the coldest white temperature (``"white_min"``), which for the
    default egg is the coldest point of the yolk surface (the top of the yolk
    on the axis), with the window [63, 70] °C: the white has set everywhere
    (63 °C) and the coldest point of the yolk surface has not yet reached
    70 °C.  That is a statement about one point only: most of the yolk
    surface and a quarter to two fifths of the yolk volume are already above
    70 °C inside the window (notebook 04 gives the numbers).  The probe
    temperature of each Monte Carlo path comes from the similarity-scaled
    response for that path's kitchen (starting temperature, bath
    temperature, size), see :class:`KitchenUncertainty`.

    The "price" is the undiscounted expected payoff, ``notional`` times the
    probability of landing in the corridor.  It is a culinary expectation,
    not a no-arbitrage price: there is no traded underlying and no discounting.

    Selection bias.  Choosing the best pull time and reporting its price on
    the same paths overstates the price (the maximum over many noisy
    estimates).  ``t_opt_s`` is chosen on the main path set; the value
    reported for it, ``price_at_opt``, is evaluated on a second, independent
    path set.

    Parameters
    ----------
    cook_times_s : array_like
        Strictly increasing, non-empty pull times [s].
    response : ThermalResponse
        Heating response of the probe.
    K_low_C, K_high_C : float
        Corridor ends [°C]; ``K_low_C < K_high_C``.
    notional : float
        Payoff inside the corridor.
    uncertainty : KitchenUncertainty, optional
        Default ``KitchenUncertainty()``.
    n_paths : int
        Paths in each of the two independent path sets.
    seed : int
        Seed; the two path sets use independent streams spawned from it.

    Returns
    -------
    dict
        - ``"cook_times_s"``: the pull times
        - ``"prices"``: expected payoff at each pull time (main paths)
        - ``"price_se"``: binomial (Monte Carlo) standard error of each
          price.  It does not include the systematic error of the similarity
          scaling of the response, which for the default egg is about five
          times larger (notebook 04).
        - ``"prob_in_corridor"``, ``"prob_undercooked"`` (probe below
          ``K_low_C``), ``"prob_overcooked"`` (above ``K_high_C``)
        - ``"t_opt_s"``: pull time with the highest price on the main paths
        - ``"price_at_opt"``, ``"price_at_opt_se"``: price and standard
          error at ``t_opt_s`` on the independent paths (unbiased)
        - ``"T_samples_at_opt_C"``: probe temperatures [°C] of the main paths at ``t_opt_s``
    """
    _check_response(response)
    t = np.asarray(cook_times_s, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("cook_times_s must be a non-empty 1-D array")
    if not np.all(np.isfinite(t)) or t[0] < 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("cook_times_s must be finite, non-negative and strictly increasing")
    if not (np.isfinite(K_low_C) and np.isfinite(K_high_C) and K_low_C < K_high_C):
        raise ValueError("need K_low_C < K_high_C")
    if not (np.isfinite(notional) and notional >= 0.0):
        raise ValueError("notional must be finite and non-negative")
    n_paths = _check_path_count("n_paths", n_paths)
    uncertainty = KitchenUncertainty() if uncertainty is None else uncertainty

    rng_main, rng_oos = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
    T_C = _simulate_probe_C(response, t, uncertainty.sample(n_paths, rng_main))

    undercooked = T_C < K_low_C
    overcooked = T_C > K_high_C
    prob_under = np.mean(undercooked, axis=0)
    prob_over = np.mean(overcooked, axis=0)
    prob_in = np.mean(~(undercooked | overcooked), axis=0)
    prices = notional * prob_in
    price_se = notional * np.sqrt(prob_in * (1.0 - prob_in) / n_paths)

    idx_opt = int(np.argmax(prices))
    t_opt = float(t[idx_opt])

    kitchens_oos = uncertainty.sample(n_paths, rng_oos)
    T_oos = _simulate_probe_C(response, np.array([t_opt]), kitchens_oos)[:, 0]
    p_oos = float(np.mean((T_oos >= K_low_C) & (T_oos <= K_high_C)))

    return {
        "cook_times_s": t,
        "prices": prices,
        "price_se": price_se,
        "prob_in_corridor": prob_in,
        "prob_undercooked": prob_under,
        "prob_overcooked": prob_over,
        "t_opt_s": t_opt,
        "price_at_opt": notional * p_oos,
        "price_at_opt_se": notional * float(np.sqrt(p_oos * (1.0 - p_oos) / n_paths)),
        "T_samples_at_opt_C": T_C[:, idx_opt],
    }


# ---------------------------------------------------------------------------
# 3. Optimal stopping ("When to Pull the Egg")
# ---------------------------------------------------------------------------

def egg_culinary_payoff(T_C, T_target_C: float = 66.5, width_C: float = 3.5, max_payoff: float = 100.0):
    """Culinary utility payoff g(T) for pulling an egg at probe temperature T_C [°C].

    A quadratic 'butterfly/tent' utility centered at ``T_target_C``:
        g(T) = max_payoff * max(1 - ((T - T_target) / width)², 0)
    With the defaults the payoff is positive for 63 °C < T < 70 °C, the same
    window as the corridor option (white set, coldest point of the yolk
    surface below 70 °C), and peaks at 66.5 °C.
    """
    if not (np.isfinite(width_C) and width_C > 0.0):
        raise ValueError("width_C must be positive")
    z = (np.asarray(T_C, dtype=float) - T_target_C) / width_C
    return max_payoff * np.maximum(1.0 - z**2, 0.0)


# The regression window (the analogue of the "in the money" paths of
# Longstaff-Schwartz) is the set of observed temperatures within
# ``width_C + 3 * obs_noise_C`` of the target: the true payoff is positive
# only within ``width_C`` of the target, and the reading is within 3 noise
# standard deviations of the truth with probability 99.7%.
_WINDOW_NOISE_SIGMAS = 3.0
_MIN_REGRESSION_PATHS = 50
# Step of the default threshold grid of the benchmark rule [°C].
_THRESHOLD_STEP_C = 0.02
_BOUNDARY_GRID_POINTS = 801


def _default_time_grid(response, uncertainty, T_target_C, width_C, step, n_sigma=4.0):
    """Grid [s] with step ``step`` that spans the payoff window in all kitchens.

    From the earliest time the probe can reach ``T_target - width`` (hot egg,
    full-boil bath, small egg, each ``n_sigma`` away) to the latest time it can
    reach ``T_target + width`` (cold egg, cool bath, large egg), rounded
    outward to a multiple of ``step``.
    """
    u = uncertainty
    fast = response.time_to_reach(
        T_target_C - width_C,
        _KELVIN + u.T_init_mean_C + n_sigma * u.T_init_std_C,
        _KELVIN + u.T_bath_max_C,
        1.0 - n_sigma * u.size_cv,
    )
    slow = response.time_to_reach(
        T_target_C + width_C,
        _KELVIN + u.T_init_mean_C - n_sigma * u.T_init_std_C,
        _KELVIN + u.T_bath_max_C - n_sigma * u.T_bath_shortfall_std_C,
        1.0 + n_sigma * u.size_cv,
    )
    if not (np.isfinite(fast) and np.isfinite(slow)):
        raise ValueError(
            "the response never reaches the payoff window within its grid; "
            "pass t_grid_s explicitly"
        )
    start = step * np.floor(fast / step)
    stop = step * np.ceil(slow / step)
    return np.arange(start, stop + 0.5 * step, step)


def _poly_basis(y, center, half_width, degree):
    """Polynomial basis 1, x, ..., x**degree in x = (y - center) / half_width."""
    return np.vander((np.asarray(y) - center) / half_width, degree + 1, increasing=True)


def _exercise_decision(coef_imm, coef_cont, y, center, half_width, degree):
    """Fitted rule: exercise if the estimated immediate payoff is positive and
    at least the estimated continuation value, and y is inside the window."""
    basis = _poly_basis(y, center, half_width, degree)
    g_hat = np.maximum(basis @ coef_imm, 0.0)
    c_hat = np.maximum(basis @ coef_cont, 0.0)
    inside = np.abs(np.asarray(y) - center) <= half_width
    return inside & (g_hat > 0.0) & (g_hat >= c_hat)


def _simulate_paths(response, t_grid, uncertainty, n, obs_noise_C, rng):
    """True probe temperatures and noisy observations [°C], each (n, len(t_grid))."""
    T_true = _simulate_probe_C(response, t_grid, uncertainty.sample(n, rng))
    obs = T_true.copy()
    if obs_noise_C > 0.0:
        obs += rng.normal(0.0, obs_noise_C, size=T_true.shape)
    return T_true, obs


def _fit_lsm_policy(payoff, obs, t_grid, discount_rate, center, half_width, degree):
    """Longstaff-Schwartz backward induction on the training paths.

    At each time j, the immediate payoff and the discounted continuation
    cashflow of the policy fitted for later times are both regressed on a
    polynomial of the observed temperature ``obs[:, j]``, using the paths
    whose observation lies in the window.  Returns the per-time coefficients
    (``None`` where too few paths) and the in-sample value at ``t_grid[0]``.
    """
    n, m = payoff.shape
    step_df = np.exp(-discount_rate * np.diff(t_grid))
    cashflow = payoff[:, -1].copy()
    policy = [None] * (m - 1)
    for j in range(m - 2, -1, -1):
        cashflow = cashflow * step_df[j]
        idx = np.nonzero(np.abs(obs[:, j] - center) <= half_width)[0]
        if len(idx) < _MIN_REGRESSION_PATHS:
            continue
        basis = _poly_basis(obs[idx, j], center, half_width, degree)
        coef_imm = np.linalg.lstsq(basis, payoff[idx, j], rcond=None)[0]
        coef_cont = np.linalg.lstsq(basis, cashflow[idx], rcond=None)[0]
        policy[j] = (coef_imm, coef_cont)
        exercise = _exercise_decision(
            coef_imm, coef_cont, obs[idx, j], center, half_width, degree
        )
        cashflow[idx[exercise]] = payoff[idx[exercise], j]
    return policy, float(np.mean(cashflow))


def _apply_policy(policy, obs, center, half_width, degree):
    """Index of the stopping time on each path under the fitted rule."""
    n, m = obs.shape
    stop = np.full(n, m - 1, dtype=int)
    active = np.ones(n, dtype=bool)
    for j in range(m - 1):
        if policy[j] is None:
            continue
        idx = np.nonzero(active)[0]
        exercise = _exercise_decision(
            policy[j][0], policy[j][1], obs[idx, j], center, half_width, degree
        )
        stop[idx[exercise]] = j
        active[idx[exercise]] = False
    return stop


def _threshold_stop(obs, threshold):
    """Index of the first time the observation reaches the threshold (else the last)."""
    reached = obs >= threshold
    return np.where(reached.any(axis=1), reached.argmax(axis=1), obs.shape[1] - 1)


def solve_american_egg_option(
    response,
    t_grid_s=None,
    T_target_C: float = 66.5,
    width_C: float = 3.5,
    max_payoff: float = 100.0,
    discount_rate: float = 0.0,
    obs_noise_C: float = 0.5,
    uncertainty=None,
    n_paths: int = 20000,
    n_paths_test: int = 20000,
    seed: int = 42,
    poly_degree: int = 3,
    thresholds_C=None,
):
    """Optimal stopping ("when to pull the egg") by Longstaff-Schwartz LSM.

    Setting.  A cook may pull the egg at any time on a grid.  The payoff is
    :func:`egg_culinary_payoff` of the *true* probe temperature at the pull
    time (default: coldest white temperature, i.e. the coldest point of the
    yolk surface, payoff positive between 63 and 70 °C).  The true probe temperature of
    each path is the deterministic FEM-scaled curve for that path's kitchen
    (starting temperature, bath temperature, size; see
    :class:`KitchenUncertainty`), so it never cools in boiling water.  The
    cook cannot see the kitchen parameters; she reads an in-egg thermometer
    that adds independent noise (``obs_noise_C``) to the true temperature and
    decides from that reading.  Utility is on a 0-100 scale; the default
    discount rate is 0.

    Method.  At each grid time the expected immediate payoff and the expected
    continuation value (the discounted payoff of the policy fitted for later
    times) are both regressed on a cubic polynomial of the observed
    temperature, using the training paths whose reading lies within
    ``width_C + 3 * obs_noise_C`` of the target (the Longstaff-Schwartz
    "in the money" paths, widened for the reading noise).  The rule exercises
    when the reading is in that window and the estimated immediate payoff is
    positive and at least the estimated continuation value.  Only the current
    reading is used (a rule that filters the reading history is not tried
    here), so the fitted policy is a lower bound for what a cook with this
    thermometer can achieve.  The rule is
    fitted on ``n_paths`` training paths and evaluated on ``n_paths_test``
    independent test paths, so the reported value has no in-sample
    (foresight) bias.  Discounting uses the actual gaps between grid times,
    so non-uniform grids are valid.

    What the value measures.  The difference between the dynamic policy and
    the best fixed timer is the value of the in-egg thermometer (information
    about the kitchen).  It is not an "early exercise premium" in the
    textbook sense (American minus European on the same expiry).  The
    perfect-foresight value (pull at the best moment of each path) is an
    upper bound.  Fixed timers and thresholds are selected on the
    training paths and evaluated on the test paths.  The one-parameter
    threshold rule is a second feasible policy; with the fine default grid
    it beats the fitted LSM rule (notebook 04), so the LSM value is not the
    best value attainable from the reading.

    Parameters
    ----------
    response : ThermalResponse
        Probe heating response.
    t_grid_s : array_like, optional
        Strictly increasing decision times [s], at least 2.  Default: 5 s
        steps from the earliest time the probe can reach ``T_target - width``
        to the latest time it can reach ``T_target + width`` across kitchens
        4 standard deviations from the mean (for the default egg's
        ``white_min`` response and default uncertainty this is the soft
        window, 240 to 580 s).
    T_target_C, width_C, max_payoff : float
        Payoff parameters, see :func:`egg_culinary_payoff`.
    discount_rate : float
        Continuous discount rate [1/s] (default 0: no impatience).
    obs_noise_C : float
        Standard deviation of the thermometer reading noise [°C], iid
        across times.  0 gives a noise-free reading.
    uncertainty : KitchenUncertainty, optional
        Default ``KitchenUncertainty()``.
    n_paths, n_paths_test : int
        Number of training and independent test paths.
    seed : int
        Seed; training and test paths use independent streams.
    poly_degree : int
        Degree of the regression polynomial (default 3).
    thresholds_C : array_like, optional
        Thresholds of the benchmark rule "pull at the first reading at or
        above the threshold".  Default: ``T_target - width`` to
        ``T_target + width`` in 0.02 °C steps.  The grid must be fine: the
        fitted rule has a continuous boundary, and on a 0.5 °C grid the
        threshold rule loses about 0.2 points of 100 against its own optimum
        (notebook 04), which turns a comparison with the LSM rule around.

    Returns
    -------
    dict
        - ``"t_grid_s"``, ``"decision_interval_s"`` (the step if the grid is
          uniform, else ``None``)
        - ``"american_value"``, ``"american_value_se"``: value of the fitted
          policy on the test paths, discounted to ``t_grid_s[0]``
        - ``"american_value_in_sample"``: the same on the training paths
        - ``"fixed_timer_values"``: expected discounted payoff of pulling at
          each grid time (test paths)
        - ``"best_fixed_time_s"``: best fixed time (chosen on the training
          paths); ``"best_fixed_value"``, ``"best_fixed_value_se"``: its
          value on the test paths
        - ``"perfect_foresight_value"``, ``"perfect_foresight_value_se"``:
          expected maximum discounted payoff along each path (upper bound)
        - ``"value_of_information"`` = ``american_value - best_fixed_value``
          and ``"value_of_information_se"`` (paired standard error)
        - ``"stopping_times_s"``, ``"stopping_temps_C"``: stopping time and
          the true probe temperature at it, for each test path
        - ``"exercise_boundary_C"``: for each grid time, the lowest observed
          temperature at which the fitted rule exercises; ``nan`` where it
          never exercises and at the last time (forced exercise)
        - ``"threshold_rule"``: dict with ``"thresholds_C"``, ``"values"``,
          ``"values_se"`` (test paths), ``"best_threshold_C"`` (chosen on
          the training paths), ``"best_value"``, ``"best_value_se"``
        - ``"T_paths_C"``, ``"T_obs_paths_C"``: true and observed
          temperatures of the first 100 test paths (for plots)
    """
    _check_response(response)
    uncertainty = KitchenUncertainty() if uncertainty is None else uncertainty
    n_paths = _check_path_count("n_paths", n_paths)
    n_paths_test = _check_path_count("n_paths_test", n_paths_test)
    if not (np.isfinite(width_C) and width_C > 0.0):
        raise ValueError("width_C must be positive")
    if not (np.isfinite(max_payoff) and max_payoff > 0.0):
        raise ValueError("max_payoff must be positive")
    if not (np.isfinite(discount_rate) and discount_rate >= 0.0):
        raise ValueError("discount_rate must be finite and non-negative")
    if not (np.isfinite(obs_noise_C) and obs_noise_C >= 0.0):
        raise ValueError("obs_noise_C must be finite and non-negative")
    if int(poly_degree) != poly_degree or poly_degree < 0:
        raise ValueError("poly_degree must be a non-negative integer")
    poly_degree = int(poly_degree)

    if t_grid_s is None:
        t_grid = _default_time_grid(response, uncertainty, T_target_C, width_C, step=5.0)
    else:
        t_grid = np.asarray(t_grid_s, dtype=float)
    if t_grid.ndim != 1 or len(t_grid) < 2:
        raise ValueError("t_grid_s must be a 1-D array with at least 2 points")
    if not np.all(np.isfinite(t_grid)) or t_grid[0] < 0.0 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid_s must be finite, non-negative and strictly increasing")
    gaps = np.diff(t_grid)
    uniform = bool(np.allclose(gaps, gaps[0], rtol=1e-9, atol=1e-9))

    if thresholds_C is None:
        thresholds = np.arange(T_target_C - width_C, T_target_C + width_C + 1e-9, _THRESHOLD_STEP_C)
    else:
        thresholds = np.asarray(thresholds_C, dtype=float)
        if thresholds.ndim != 1 or len(thresholds) == 0:
            raise ValueError("thresholds_C must be a non-empty 1-D array")

    center = float(T_target_C)
    half_width = width_C + _WINDOW_NOISE_SIGMAS * obs_noise_C
    discount = np.exp(-discount_rate * (t_grid - t_grid[0]))

    rng_train, rng_test = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
    T_train, obs_train = _simulate_paths(response, t_grid, uncertainty, n_paths, obs_noise_C, rng_train)
    T_test, obs_test = _simulate_paths(response, t_grid, uncertainty, n_paths_test, obs_noise_C, rng_test)
    pay_train = egg_culinary_payoff(T_train, T_target_C, width_C, max_payoff)
    pay_test = egg_culinary_payoff(T_test, T_target_C, width_C, max_payoff)
    disc_pay_test = pay_test * discount[None, :]
    rows = np.arange(n_paths_test)

    def mean_se(x):
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

    # Dynamic policy: fit on training paths, evaluate on test paths.
    policy, value_in_sample = _fit_lsm_policy(
        pay_train, obs_train, t_grid, discount_rate, center, half_width, poly_degree
    )
    stop = _apply_policy(policy, obs_test, center, half_width, poly_degree)
    value_paths = disc_pay_test[rows, stop]
    american_value, american_se = mean_se(value_paths)

    # Fixed timers: time chosen on training paths, values on test paths.
    fixed_values = np.mean(disc_pay_test, axis=0)
    best_idx = int(np.argmax(np.mean(pay_train * discount[None, :], axis=0)))
    best_paths = disc_pay_test[:, best_idx]
    best_fixed_value, best_fixed_se = mean_se(best_paths)

    # Perfect foresight (pathwise best pull time): an upper bound.
    foresight_value, foresight_se = mean_se(np.max(disc_pay_test, axis=1))

    voi_value, voi_se = mean_se(value_paths - best_paths)

    # Threshold rules: threshold chosen on training paths, values on test paths.
    thr_values = np.empty(len(thresholds))
    thr_se = np.empty(len(thresholds))
    thr_train = np.empty(len(thresholds))
    for i, thr in enumerate(thresholds):
        thr_values[i], thr_se[i] = mean_se(disc_pay_test[rows, _threshold_stop(obs_test, thr)])
        thr_train[i] = np.mean(
            (pay_train * discount[None, :])[np.arange(n_paths), _threshold_stop(obs_train, thr)]
        )
    best_thr = int(np.argmax(thr_train))

    # Exercise boundary of the fitted rule: lowest exercising reading.
    y_fine = np.linspace(center - half_width, center + half_width, _BOUNDARY_GRID_POINTS)
    boundary = np.full(len(t_grid), np.nan)
    for j in range(len(t_grid) - 1):
        if policy[j] is None:
            continue
        exercise = _exercise_decision(
            policy[j][0], policy[j][1], y_fine, center, half_width, poly_degree
        )
        if exercise.any():
            boundary[j] = float(y_fine[np.argmax(exercise)])

    return {
        "t_grid_s": t_grid,
        "decision_interval_s": float(gaps[0]) if uniform else None,
        "american_value": american_value,
        "american_value_se": american_se,
        "american_value_in_sample": value_in_sample,
        "fixed_timer_values": fixed_values,
        "best_fixed_time_s": float(t_grid[best_idx]),
        "best_fixed_value": best_fixed_value,
        "best_fixed_value_se": best_fixed_se,
        "perfect_foresight_value": foresight_value,
        "perfect_foresight_value_se": foresight_se,
        "value_of_information": american_value - best_fixed_value,
        "value_of_information_se": voi_se,
        "stopping_times_s": t_grid[stop],
        "stopping_temps_C": T_test[rows, stop],
        "exercise_boundary_C": boundary,
        "threshold_rule": {
            "thresholds_C": thresholds,
            "values": thr_values,
            "values_se": thr_se,
            "best_threshold_C": float(thresholds[best_thr]),
            "best_value": float(thr_values[best_thr]),
            "best_value_se": float(thr_se[best_thr]),
        },
        "T_paths_C": T_test[:100],
        "T_obs_paths_C": obs_test[:100],
    }
