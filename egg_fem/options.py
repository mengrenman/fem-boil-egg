"""Quantitative finance and stochastic optimal stopping on the egg heat equation.

Bridges the axisymmetric FEM heat equation with options pricing theory:
  1. Black-Scholes <-> Heat Equation Duality (1D FEM solver vs closed-form Black-Scholes)
  2. Corridor / Digital Option Pricing on Egg Doneness under Kitchen Uncertainty
  3. American Option / Optimal Stopping ("When to Pull the Egg") via Longstaff-Schwartz
"""

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


# ---------------------------------------------------------------------------
# 1. Black-Scholes <-> 1D Heat Equation FEM Duality
# ---------------------------------------------------------------------------

def black_scholes_call(S, K: float, T: float, r: float, sigma: float):
    """Exact Black-Scholes formula for a European call option."""
    S_arr = np.asarray(S, dtype=float)
    if T <= 0.0:
        return np.maximum(S_arr - K, 0.0)
    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S_arr * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


@BilinearForm
def _mass_1d(u, v, w):
    return u * v


@BilinearForm
def _stiff_1d(u, v, w):
    return dot(grad(u), grad(v))


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

    Returns
    -------
    result : dict
        - ``"S"``: stock price grid
        - ``"V_fem"``: FEM option value at t = 0
        - ``"V_exact"``: closed-form Black-Scholes call price at t = 0
        - ``"x"``: log-moneyness grid x = ln(S/K)
        - ``"u_init"``: transformed heat initial condition u(x, 0)
        - ``"u_final"``: diffused heat solution u(x, τ_max)
    """
    x_grid = np.linspace(x_min, x_max, n_nodes)
    mesh = MeshLine(x_grid)
    basis = Basis(mesh, ElementLineP1())

    k = 2.0 * r / (sigma**2)
    alpha = -0.5 * (k - 1.0)
    beta = -0.25 * (k + 1.0) ** 2

    tau_max = 0.5 * (sigma**2) * T
    dtau = tau_max / n_steps

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
# 2. Egg Thermal Surrogate & Kitchen Uncertainty Monte Carlo
# ---------------------------------------------------------------------------

def fourier_sphere_center_temp(
    t,
    T_init: float = 293.15,
    T_bath: float = 373.15,
    R_eff: float = 0.0228,
    alpha_diff: float = 1.32e-7,
    n_terms: int = 15,
):
    """Analytical Fourier-series center temperature [K] for an equivalent sphere.

    Calibrated to match the multi-region axisymmetric FEM center trajectory
    (default R_eff = 22.8 mm, alpha_diff = 1.32e-7 m²/s gives soft-boiled 63 °C
    at t ≈ 592 s from 20 °C start).
    """
    t_arr = np.maximum(np.asarray(t, dtype=float), 0.0)
    series = np.zeros_like(t_arr)
    for n in range(1, n_terms + 1):
        sign = 1.0 if (n % 2 == 1) else -1.0
        decay = np.exp(-(n * np.pi / R_eff) ** 2 * alpha_diff * t_arr)
        series = series + sign * decay
    T_center = T_bath + 2.0 * (T_init - T_bath) * series
    # Clamp slight numerical truncation overshoot at t = 0
    return np.clip(T_center, np.minimum(T_init, T_bath), np.maximum(T_init, T_bath))


def price_egg_corridor_option(
    cook_times_s,
    K_low_C: float = 62.5,
    K_high_C: float = 65.5,
    notional: float = 10.0,
    T_init_mean_C: float = 20.0,
    T_init_std_C: float = 3.5,
    T_bath_mean_C: float = 99.0,
    T_bath_std_C: float = 1.2,
    R_eff_mean: float = 0.0228,
    R_eff_std: float = 0.0014,
    n_paths: int = 5000,
    seed: int = 42,
):
    """Price a Binary Corridor Option ("Perfect Soft-Boil Warranty") across cook times.

    Payoff at maturity t_cook:
        Payoff = notional   if  K_low_C <= T_center(t_cook) <= K_high_C
                 0          otherwise

    Propagates kitchen parameter uncertainties (initial egg temperature,
    effective egg radius/size, and water bath temperature) through the thermal
    diffusion model.

    Returns
    -------
    result : dict
        - ``"cook_times_s"``: array of pull times [s]
        - ``"prices"``: expected option payoff [$] at each pull time
        - ``"prob_in_corridor"``: probability of hitting the soft-boil window
        - ``"prob_undercooked"``: probability T_center < K_low_C
        - ``"prob_overcooked"``: probability T_center > K_high_C
        - ``"T_samples_at_opt"``: distribution of center temperatures [°C] at optimal pull time
        - ``"t_opt_s"``: pull time [s] that maximizes corridor option value
    """
    rng = np.random.default_rng(seed)
    t_grid = np.asarray(cook_times_s, dtype=float)

    T_init_samples = 273.15 + rng.normal(T_init_mean_C, T_init_std_C, size=(n_paths, 1))
    T_bath_samples = np.clip(
        273.15 + rng.normal(T_bath_mean_C, T_bath_std_C, size=(n_paths, 1)),
        355.15,
        373.15,
    )
    R_samples = np.clip(
        rng.normal(R_eff_mean, R_eff_std, size=(n_paths, 1)),
        0.016,
        0.030,
    )

    # Evaluate center temperature [°C] for all paths across t_grid: shape (n_paths, len(t_grid))
    T_center_K = fourier_sphere_center_temp(
        t_grid[None, :],
        T_init=T_init_samples,
        T_bath=T_bath_samples,
        R_eff=R_samples,
    )
    T_center_C = T_center_K - 273.15

    undercooked = T_center_C < K_low_C
    overcooked = T_center_C > K_high_C
    in_corridor = (~undercooked) & (~overcooked)

    prob_in = np.mean(in_corridor, axis=0)
    prob_under = np.mean(undercooked, axis=0)
    prob_over = np.mean(overcooked, axis=0)
    prices = notional * prob_in

    idx_opt = int(np.argmax(prices))

    return {
        "cook_times_s": t_grid,
        "prices": prices,
        "prob_in_corridor": prob_in,
        "prob_undercooked": prob_under,
        "prob_overcooked": prob_over,
        "T_samples_at_opt": T_center_C[:, idx_opt],
        "t_opt_s": float(t_grid[idx_opt]),
    }


# ---------------------------------------------------------------------------
# 3. American Option / Optimal Stopping ("When to Pull the Egg")
# ---------------------------------------------------------------------------

def egg_culinary_payoff(T_C, T_target_C: float = 64.0, width_C: float = 4.5, max_payoff: float = 100.0):
    """Culinary utility payoff g(T) for pulling an egg at center temperature T_C [°C].

    Follows a quadratic 'butterfly/tent' utility profile centered at T_target_C:
        g(T) = max_payoff * max(1 - ((T - T_target) / width)², 0)
    """
    z = (np.asarray(T_C, dtype=float) - T_target_C) / width_C
    return max_payoff * np.maximum(1.0 - z**2, 0.0)


def solve_american_egg_option(
    t_grid_s=None,
    T_target_C: float = 64.0,
    width_C: float = 4.5,
    max_payoff: float = 100.0,
    discount_rate: float = 0.001,  # impatience / hunger discount rate [1/s]
    sigma_temp_C: float = 1.2,     # thermal volatility [°C / sqrt(min)]
    n_paths: int = 4000,
    seed: int = 42,
):
    """Solve the American Optimal Stopping problem ("When to Pull the Egg")
    using Longstaff-Schwartz Least-Squares Monte Carlo (LSM).

    Simulates stochastic center-temperature trajectories driven by the deterministic
    FEM heating drift plus thermal fluctuations (stove bubbling, convective eddies,
    size variance), and computes the backward-inducted continuation value vs.
    immediate exercise payoff at every decision time.

    Returns
    -------
    result : dict
        - ``"t_grid_s"``: time grid [s]
        - ``"T_paths_C"``: simulated temperature paths [°C], shape ``(n_paths, n_steps)``
        - ``"american_value"``: optimal stopping expected discounted payoff
        - ``"european_values"``: expected discounted payoff for fixed pull times
        - ``"stopping_times_s"``: optimal pull time [s] for each path
        - ``"stopping_temps_C"``: center temperature [°C] at the moment each path is pulled
        - ``"exercise_boundary_C"``: estimated lower exercise threshold T*(t) [°C]
    """
    if t_grid_s is None:
        t_grid_s = np.linspace(300.0, 780.0, 97)  # 5 min to 13 min every 5 s
    t_grid = np.asarray(t_grid_s, dtype=float)
    n_steps = len(t_grid)
    dt = float(t_grid[1] - t_grid[0])

    rng = np.random.default_rng(seed)

    # Base deterministic curve from FEM/Fourier sphere
    T_det_C = fourier_sphere_center_temp(t_grid) - 273.15
    dT_det = np.diff(T_det_C)

    # Simulate stochastic paths around the deterministic drift
    T_paths = np.empty((n_paths, n_steps))
    T_paths[:, 0] = T_det_C[0] + rng.normal(0.0, 1.8, size=n_paths)
    rate_mult = np.clip(rng.normal(1.0, 0.08, size=n_paths), 0.75, 1.25)

    vol_step = sigma_temp_C * np.sqrt(dt / 60.0)
    for j in range(n_steps - 1):
        dW = rng.normal(0.0, vol_step, size=n_paths)
        # Temperature increments are positive on average with stochastic diffusion
        T_paths[:, j + 1] = T_paths[:, j] + rate_mult * dT_det[j] + dW

    # Immediate exercise payoffs g(T_t) at all times
    payoffs = egg_culinary_payoff(
        T_paths, T_target_C=T_target_C, width_C=width_C, max_payoff=max_payoff
    )

    # European option values for fixed pull times (discounted back to t_grid[0])
    discounts = np.exp(-discount_rate * (t_grid - t_grid[0]))
    european_values = np.mean(payoffs * discounts[None, :], axis=0)

    # Longstaff-Schwartz backward induction
    cashflow = payoffs[:, -1].copy()
    stop_idx = np.full(n_paths, n_steps - 1, dtype=int)
    exercise_boundary = np.full(n_steps, np.nan)
    exercise_boundary[-1] = T_target_C
    df_step = np.exp(-discount_rate * dt)

    for j in range(n_steps - 2, -1, -1):
        cashflow = cashflow * df_step
        in_the_money = payoffs[:, j] > 0.0

        if np.count_nonzero(in_the_money) > 15:
            X = T_paths[in_the_money, j]
            Y = cashflow[in_the_money]
            # Degree-3 polynomial basis in normalized temperature
            X_norm = (X - T_target_C) / width_C
            design = np.column_stack([np.ones_like(X_norm), X_norm, X_norm**2, X_norm**3])
            coeffs, *_ = np.linalg.lstsq(design, Y, rcond=None)
            continuation = np.maximum(design @ coeffs, 0.0)

            exercise_now = payoffs[in_the_money, j] >= continuation
            itm_indices = np.where(in_the_money)[0]
            ex_indices = itm_indices[exercise_now]

            if len(ex_indices) > 0:
                cashflow[ex_indices] = payoffs[ex_indices, j]
                stop_idx[ex_indices] = j
                exercise_boundary[j] = float(np.percentile(T_paths[ex_indices, j], 15))
            else:
                exercise_boundary[j] = np.nan
        else:
            exercise_boundary[j] = np.nan

    american_value = float(np.mean(cashflow))
    stopping_times_s = t_grid[stop_idx]
    stopping_temps_C = T_paths[np.arange(n_paths), stop_idx]

    return {
        "t_grid_s": t_grid,
        "T_paths_C": T_paths,
        "american_value": american_value,
        "european_values": european_values,
        "stopping_times_s": stopping_times_s,
        "stopping_temps_C": stopping_temps_C,
        "exercise_boundary_C": exercise_boundary,
    }
