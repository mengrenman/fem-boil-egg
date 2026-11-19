"""Tests for the options-on-an-egg finance module (egg_fem.options).

The FEM core (materials, geometry, solver, doneness) is tested in
test_fem.py.  Most Monte Carlo mechanics are tested on the fast analytic
sphere response; the FEM-driven response of the default egg is built once
per module and used for the similarity check and the pinned headline numbers.
"""

import warnings

import numpy as np
import pytest

from egg_fem import (
    KitchenUncertainty,
    ThermalResponse,
    T_BOILING,
    T_ROOM,
    HeatSolver,
    black_scholes_call,
    check_similarity_against_fem,
    create_circular_mesh,
    create_egg_mesh,
    doneness_times,
    egg_culinary_payoff,
    egg_white_properties,
    egg_yolk_properties,
    fourier_sphere_center_temp,
    price_egg_corridor_option,
    solve_american_egg_option,
    solve_black_scholes_heat_fem,
)

K0 = 273.15  # 0 degC in kelvin


@pytest.fixture(scope="module")
def sphere_response():
    """Exact center response of a 22.8 mm homogeneous sphere (fast)."""
    return ThermalResponse.from_sphere(0.0228, 1.32e-7, 1500.0, 1501)


@pytest.fixture(scope="module")
def fem_response():
    """White-min response of the default egg from one FEM run (about 4 s)."""
    return ThermalResponse.from_fem()


# ---------------------------------------------------------------------------
# 1. Black-Scholes <-> heat equation
# ---------------------------------------------------------------------------

def _bs_node_error(n_nodes, n_steps, **kwargs):
    res = solve_black_scholes_heat_fem(n_nodes=n_nodes, n_steps=n_steps, **kwargs)
    S = res["S"]
    mask = (S >= 50.0) & (S <= 150.0)
    return float(np.max(np.abs(res["V_fem"][mask] - res["V_exact"][mask])))


def test_black_scholes_closed_form_textbook_value():
    # Hull-style check: S = K = 100, T = 1, r = 5%, sigma = 20% -> 10.4506.
    assert float(black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)) == pytest.approx(
        10.4506, abs=5e-5
    )
    # At expiry the call is its intrinsic value.
    assert np.allclose(black_scholes_call([80.0, 100.0, 130.0], 100.0, 0.0, 0.05, 0.2),
                       [0.0, 0.0, 30.0])
    # S = 0 is worth zero and produces no warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert float(black_scholes_call(0.0, 100.0, 1.0, 0.05, 0.2)) == 0.0


def test_black_scholes_heat_fem_accuracy_at_two_step_counts():
    """Nodal max error on S in [50, 150] (401 nodes) is tight and first order in time."""
    err_200 = _bs_node_error(401, 200)
    err_400 = _bs_node_error(401, 400)
    # Measured 0.00591 and 0.00354 USD; the bounds fail on any real change.
    assert 0.0050 < err_200 < 0.0066
    assert 0.0030 < err_400 < 0.0040
    # Backward Euler: halving the step cuts the (time-dominated) error by well
    # under a factor 2 here because the space error (0.0016) does not move.
    assert 1.4 < err_200 / err_400 < 2.0
    # With many more steps the error approaches the spatial P1 error.
    assert _bs_node_error(401, 3200) < 0.0017


def test_black_scholes_heat_fem_other_parameters():
    """The transform is right beyond the default parameters (negative rate, high rate)."""
    S_err = _bs_node_error(801, 400, r=-0.02, sigma=0.3, K=80.0)
    assert S_err < 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0},
        {"sigma": -0.2},
        {"T": 0.0},
        {"T": -1.0},
        {"K": 0.0},
        {"x_min": 0.0},
        {"x_min": 1.0},
        {"x_max": 0.0},
        {"x_max": -1.0},
        {"n_nodes": 2},
        {"n_steps": 0},
        {"sigma": float("nan")},
    ],
)
def test_black_scholes_heat_fem_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        solve_black_scholes_heat_fem(n_nodes=kwargs.pop("n_nodes", 101), n_steps=kwargs.pop("n_steps", 10), **kwargs)


@pytest.mark.parametrize(
    "args",
    [
        (100.0, 0.0, 1.0, 0.05, 0.2),    # K = 0
        (100.0, 100.0, -1.0, 0.05, 0.2),  # T < 0
        (100.0, 100.0, 1.0, 0.05, 0.0),  # sigma = 0
        (100.0, 100.0, 1.0, 0.05, -0.1),  # sigma < 0
        (-5.0, 100.0, 1.0, 0.05, 0.2),   # S < 0
    ],
)
def test_black_scholes_call_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        black_scholes_call(*args)


def test_black_scholes_heat_fem_small_sigma_raises_instead_of_garbage():
    """Small sigma means huge k = 2r/sigma^2: exponents overflow or wipe out precision."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow warnings leak out
        for sigma in (0.04, 0.02, 0.01, 0.001):
            with pytest.raises(ValueError, match="too large"):
                solve_black_scholes_heat_fem(sigma=sigma, n_nodes=101, n_steps=10)
        # A narrower domain brings the same sigma back into the safe range ...
        res = solve_black_scholes_heat_fem(
            sigma=0.05, x_min=-0.5, x_max=0.5, n_nodes=401, n_steps=400
        )
        S = res["S"]
        mask = (S >= 70.0) & (S <= 130.0)
        assert np.all(np.isfinite(res["V_fem"]))
        assert np.max(np.abs(res["V_fem"][mask] - res["V_exact"][mask])) < 0.02
        # ... and the default sigma is untouched by the guard.
        assert _bs_node_error(101, 50) < 0.1


# ---------------------------------------------------------------------------
# 2a. Exact sphere center temperature
# ---------------------------------------------------------------------------

def _sphere_theta_reference(fo, n_terms=2000):
    """Independent reference for Fo >= 0.02: alternating series, many terms."""
    n = np.arange(1, n_terms + 1)
    sign = np.where(n % 2 == 1, 1.0, -1.0)
    return 2.0 * np.sum(sign[:, None] * np.exp(-(n[:, None] * np.pi) ** 2 * fo[None, :]), axis=0)


@pytest.mark.parametrize("n_terms", [1, 2, 3, 14, 15, 16, None])
def test_sphere_center_is_exactly_t_init_at_time_zero(n_terms):
    # Independent of the parity of n_terms (the old odd-n_terms-plus-clip trick).
    T = fourier_sphere_center_temp(np.array([0.0, -5.0]), T_init=T_ROOM, T_bath=T_BOILING,
                                   n_terms=n_terms)
    assert T[0] == T_ROOM
    assert T[1] == T_ROOM  # negative times are treated as 0
    assert fourier_sphere_center_temp(0.0, T_init=281.15, T_bath=373.15,
                                      n_terms=n_terms) == 281.15


@pytest.mark.parametrize("n_terms", [10, 11, 12, 15, 16, None])
def test_sphere_center_matches_independent_series_for_any_n_terms_parity(n_terms):
    R, alpha = 0.0228, 1.32e-7
    t_s = np.array([30.0, 40.0, 60.0, 120.0, 300.0, 580.0, 900.0, 1800.0])  # Fo >= 0.0076
    fo = alpha * t_s / R**2
    T = fourier_sphere_center_temp(t_s, T_ROOM, T_BOILING, R, alpha, n_terms=n_terms)
    theta_ref = _sphere_theta_reference(fo)
    expected = T_BOILING + (T_ROOM - T_BOILING) * theta_ref
    # The reference series is itself only valid for Fo >= 0.02 (t >= 80 s here);
    # for earlier times use the leading image term, exact to e^-100.
    small = fo < 0.02
    theta_img = 1.0 - 2.0 / np.sqrt(np.pi * fo) * np.exp(-1.0 / (4.0 * fo))
    expected[small] = T_BOILING + (T_ROOM - T_BOILING) * theta_img[small]
    assert np.max(np.abs(T - expected)) < 1e-9


def test_sphere_center_small_times_are_physical():
    """The center cannot know about the surface yet: T = T_init, bounded, monotone."""
    t_s = np.array([0.0, 1e-9, 1e-3, 1.0, 5.0, 10.0, 16.0, 20.0, 40.0, 80.0])
    T = fourier_sphere_center_temp(t_s, T_ROOM, T_BOILING)
    assert np.all(T >= T_ROOM) and np.all(T <= T_BOILING)
    assert np.all(np.diff(T) >= 0.0)
    assert np.all(T[:6] == T_ROOM)  # indistinguishable from T_init to machine precision at Fo < 4e-3
    assert 0.0 < T[-1] - T_ROOM < 1.0  # Fo = 0.0058 at 40 s: first warming only at t ~ 80 s


def test_sphere_center_switch_between_series_and_image_form_is_seamless():
    # theta(Fo) computed with the two representations on either side of Fo = 0.1.
    R, alpha = 0.02, 1.0e-7
    t_switch = 0.1 * R**2 / alpha
    eps = 1e-7
    below, at, above = fourier_sphere_center_temp(
        np.array([t_switch * (1 - eps), t_switch, t_switch * (1 + eps)]),
        0.0, 1.0, R, alpha,
    )
    assert abs(above - below) < 1e-5  # slope is O(1) per unit Fo: 2e-7 * 2.6 = 5e-7
    assert abs(at - below) < 1e-5


def test_sphere_center_long_time_limit():
    """theta -> 2 exp(-pi^2 Fo): the Fourier-Bessel leading term."""
    R, alpha = 0.0228, 1.32e-7
    t_s = np.array([2400.0, 3600.0])  # Fo = 0.61, 0.91: the n = 2 term is below 1e-10
    fo = alpha * t_s / R**2
    theta = 1.0 - fourier_sphere_center_temp(t_s, 0.0, 1.0, R, alpha)  # T_init = 0, T_bath = 1
    assert np.allclose(theta, 2.0 * np.exp(-np.pi**2 * fo), rtol=1e-8)


def test_sphere_center_broadcasts_and_validates():
    t_s = np.linspace(0.0, 900.0, 7)[None, :]
    T0 = np.array([[285.0], [293.15], [300.0]])
    T = fourier_sphere_center_temp(t_s, T0, 373.15, 0.0228, 1.32e-7)
    assert T.shape == (3, 7)
    assert np.all(T[:, 0] == T0[:, 0])
    with pytest.raises(ValueError):
        fourier_sphere_center_temp(100.0, R_eff=0.0)
    with pytest.raises(ValueError):
        fourier_sphere_center_temp(100.0, alpha_diff=-1e-7)
    with pytest.raises(ValueError):
        fourier_sphere_center_temp(100.0, n_terms=0)


# ---------------------------------------------------------------------------
# 2b. ThermalResponse
# ---------------------------------------------------------------------------

def test_response_from_sphere_matches_sphere_formula_and_similarity_is_exact(sphere_response):
    t = np.linspace(0.0, 1400.0, 701)
    T = sphere_response(t, T_ROOM, T_BOILING)
    exact = fourier_sphere_center_temp(t, T_ROOM, T_BOILING, 0.0228, 1.32e-7)
    assert np.max(np.abs(T - exact)) < 5e-4  # linear interpolation on a 1 s grid
    # A sphere of radius R*s is exactly the response at time t / s^2.
    for s in (0.93, 1.07):
        t_s = np.linspace(0.0, 1200.0, 241)  # 1200 / 0.93**2 is inside the 1500 s grid
        T_s = sphere_response(t_s, 281.15, 371.0, size_scale=s)
        exact_s = fourier_sphere_center_temp(t_s, 281.15, 371.0, 0.0228 * s, 1.32e-7)
        assert np.max(np.abs(T_s - exact_s)) < 5e-4


def test_response_broadcasting_and_limits(sphere_response):
    t = np.array([[0.0, 300.0, 600.0]])
    T0 = np.array([[285.0], [293.0]])
    Tb = np.array([[373.0], [371.0]])
    s = np.array([[0.98], [1.02]])
    T = sphere_response(t, T0, Tb, s)
    assert T.shape == (2, 3)
    assert np.all(T[:, 0] == T0[:, 0])
    assert np.all(np.diff(T, axis=1) > 0.0)
    assert np.all(T < Tb)
    assert np.ndim(sphere_response(100.0, 293.15, 373.15)) == 0


def test_response_rejects_bad_calls_and_bad_construction(sphere_response):
    with pytest.raises(ValueError, match="beyond the response grid"):
        sphere_response(1500.0, T_ROOM, T_BOILING, size_scale=0.9)  # 1500/0.81 > 1500
    with pytest.raises(ValueError):
        sphere_response(-1.0, T_ROOM, T_BOILING)
    with pytest.raises(ValueError):
        sphere_response(10.0, T_ROOM, T_BOILING, size_scale=0.0)
    good_t = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        ThermalResponse(good_t, [1.0, 0.5, 0.7])  # increasing theta
    with pytest.raises(ValueError):
        ThermalResponse([1.0, 2.0, 3.0], [1.0, 0.5, 0.2])  # does not start at 0
    with pytest.raises(ValueError):
        ThermalResponse([0.0, 2.0, 1.0], [1.0, 0.5, 0.2])  # time not increasing
    with pytest.raises(ValueError):
        ThermalResponse(good_t, [1.2, 0.5, 0.2])  # theta > 1
    with pytest.raises(ValueError):
        ThermalResponse.from_fem(probe="surface")
    with pytest.raises(ValueError):
        ThermalResponse.from_sphere(0.02, 1e-7, 0.0, 10)


def test_regional_minimum_probes_need_a_heated_egg(fem_response):
    """T = T_bath + (T_init - T_bath) theta puts the minimum where theta is largest only
    when T_init < T_bath; the response refuses kitchens for which that fails."""
    assert fem_response.probe == "white_min"
    assert fem_response(300.0, T_ROOM, T_BOILING) < T_BOILING  # heating: fine
    for T_init, T_bath in ((T_BOILING, T_BOILING), (T_BOILING + 5.0, T_BOILING)):
        with pytest.raises(ValueError, match="heated egg"):
            fem_response(300.0, T_init, T_bath)
        with pytest.raises(ValueError, match="heated egg"):
            fem_response.time_to_reach(63.0, T_init, T_bath)
    # one bad kitchen among good ones is enough
    with pytest.raises(ValueError, match="heated egg"):
        fem_response(300.0, np.array([T_ROOM, T_BOILING + 1.0]), T_BOILING)
    with pytest.raises(ValueError, match="heated egg"):
        ThermalResponse.from_fem(probe="yolk_min", T_init=T_BOILING + 10.0, T_bath=T_ROOM)
    # the center and the analytic sphere response have no such restriction
    center = ThermalResponse([0.0, 1.0, 2.0], [1.0, 0.5, 0.2], probe="center")
    assert center(1.0, T_BOILING + 10.0, T_BOILING) == pytest.approx(T_BOILING + 5.0)


def test_time_to_reach_inverts_the_response(sphere_response):
    for T_target in (40.0, 63.0, 66.5, 70.0, 90.0):
        t = sphere_response.time_to_reach(T_target, T_ROOM, T_BOILING)
        assert sphere_response(t, T_ROOM, T_BOILING) - K0 == pytest.approx(T_target, abs=2e-3)
    # similarity: a larger egg takes s^2 longer to reach the same temperature
    t1 = sphere_response.time_to_reach(63.0, T_ROOM, T_BOILING, 1.0)
    t2 = sphere_response.time_to_reach(63.0, T_ROOM, T_BOILING, 1.1)
    assert t2 == pytest.approx(1.21 * t1, rel=1e-9)
    assert sphere_response.time_to_reach(10.0, T_ROOM, T_BOILING) == 0.0  # already warmer
    assert np.isnan(sphere_response.time_to_reach(99.9999, T_ROOM, T_BOILING))  # not within grid
    arr = sphere_response.time_to_reach(63.0, T_ROOM + np.array([-4.0, 0.0, 4.0]), T_BOILING)
    assert arr.shape == (3,) and np.all(np.diff(arr) < 0.0)  # a warmer egg cooks faster


def test_fem_response_is_a_valid_default_egg_response(fem_response):
    th = fem_response.theta
    assert fem_response.probe == "white_min"
    assert th[0] == 1.0 and np.all(np.diff(th) <= 0.0) and th[-1] < 0.05
    # The default-kitchen crossing times of the coldest white point (prototype:
    # 343.3 s, 373.1 s, 406.7 s; 343.3 s is also doneness_times()["soft"]).
    t63, t665, t70 = (fem_response.time_to_reach(x, T_ROOM, T_BOILING) for x in (63.0, 66.5, 70.0))
    assert t63 == pytest.approx(343.3, abs=1.0)
    assert t665 == pytest.approx(373.1, abs=1.0)
    assert t70 == pytest.approx(406.7, abs=1.0)


def test_fem_response_probes_agree_with_solver_diagnostics():
    """theta normalization and inversion reproduce doneness_times on a coarse mesh."""
    mesh = create_egg_mesh(mesh_size=0.003, mesh_size_boundary=0.0015)
    materials = {"white": egg_white_properties, "yolk": egg_yolk_properties}
    solver = HeatSolver(mesh, materials, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0)
    solver.run(800.0, progress=False)
    times, center_C = solver.center_temperature_history()

    white = ThermalResponse.from_fem(mesh=mesh, dt=5.0, t_end=800.0)
    soft = doneness_times(solver)["soft"]
    assert white.time_to_reach(63.0, T_ROOM, T_BOILING) == pytest.approx(soft, abs=0.05)

    center = ThermalResponse.from_fem(probe="center", mesh=mesh, dt=5.0, t_end=800.0)
    # (theta is clipped to <= 1 and made non-increasing; on this coarse mesh the
    # raw center temperature undershoots T_init by up to 0.02 K around 100-150 s)
    assert np.max(np.abs(center(times, T_ROOM, T_BOILING) - K0 - center_C)) < 0.03
    assert np.all(np.diff(center.theta) <= 0.0) and np.max(center.theta) == 1.0
    late = times > 200.0
    assert np.max(np.abs(center(times, T_ROOM, T_BOILING) - K0 - center_C)[late]) < 1e-6
    # the yolk center is the coldest yolk point: yolk_min == center for this egg
    yolk = ThermalResponse.from_fem(probe="yolk_min", mesh=mesh, dt=5.0, t_end=800.0)
    assert np.max(np.abs(yolk.theta - center.theta)) < 1e-3  # early undershoot differs a little
    assert np.max(np.abs(yolk.theta - center.theta)[late]) < 1e-9
    # the coldest white point reaches 63 degC long before the yolk center does
    assert white.time_to_reach(63.0, T_ROOM, T_BOILING) < center.time_to_reach(63.0, T_ROOM, T_BOILING)


def test_from_fem_needs_regions_for_region_probes():
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.004)
    with pytest.raises(ValueError):
        ThermalResponse.from_fem(probe="white_min", mesh=mesh, materials=egg_white_properties,
                                 dt=10.0, t_end=100.0)
    resp = ThermalResponse.from_fem(probe="center", mesh=mesh, materials=egg_white_properties,
                                    dt=10.0, t_end=300.0)
    assert resp.theta[0] == 1.0 and resp.theta[-1] < 0.9


def test_similarity_scaling_vs_direct_fem_at_perturbed_kitchen():
    """The scaled response tracks a direct FEM run of a cold, big egg in cooler water.

    Coarse mesh (3 mm, dt 5 s) for speed.  Measured: the scaled response is 4.2-4.5 s
    early for the 63-70 degC crossings (about 1%) and at most 1.2 degC off in
    temperature (on this coarse mesh); the perturbation itself shifts the crossings
    by about 70 s.  At full resolution the scaled crossings are 3.1-4.4 s early.
    """
    geom = dict(mesh_size=0.003, mesh_size_boundary=0.0015)
    mesh = create_egg_mesh(**geom)
    coarse = ThermalResponse.from_fem(mesh=mesh, dt=5.0, t_end=700.0)
    ref_cross = np.array([coarse.time_to_reach(x, T_ROOM, T_BOILING) for x in (63.0, 66.5, 70.0)])

    T0, Tb, s = 281.15, 371.65, 1.03  # 8 degC egg, 98.5 degC water, 3% larger
    out = check_similarity_against_fem(coarse, T0, Tb, s, geometry=geom, dt=5.0)
    shift = out["t_direct_s"] - ref_cross
    assert np.all(shift > 60.0)                       # the perturbation matters ...
    assert np.all(np.abs(out["t_diff_s"]) < 6.0)       # ... and scaling captures > 90% of it
    assert np.all(np.abs(out["t_diff_s"]) < 0.1 * shift)
    assert out["max_abs_diff_C"] < 1.5
    # Size alone is exact for constant-geometry scaling (properties are the same
    # function of T): crossing times agree to the time step.
    size_only = check_similarity_against_fem(coarse, T_ROOM, T_BOILING, 1.05, geometry=geom, dt=5.0)
    assert np.all(np.abs(size_only["t_diff_s"]) < 0.5)


# ---------------------------------------------------------------------------
# 2c. Kitchen uncertainty
# ---------------------------------------------------------------------------

def test_kitchen_uncertainty_defaults_and_bath_never_exceeds_boiling():
    unc = KitchenUncertainty()
    T0, Tb, s = unc.sample(200000, np.random.default_rng(7))
    assert T0.shape == Tb.shape == s.shape == (200000,)
    assert np.max(Tb) <= K0 + 100.0
    assert np.mean(Tb == K0 + 100.0) == 0.0  # no point mass at exactly 100 degC
    # half-normal shortfall: mean sigma*sqrt(2/pi) = 0.957, sd sigma*sqrt(1 - 2/pi) = 0.725
    shortfall = K0 + 100.0 - Tb
    assert shortfall.mean() == pytest.approx(1.2 * np.sqrt(2 / np.pi), abs=0.01)
    assert shortfall.std() == pytest.approx(1.2 * np.sqrt(1 - 2 / np.pi), abs=0.01)
    assert T0.mean() - K0 == pytest.approx(20.0, abs=0.04)
    assert T0.std() == pytest.approx(3.5, abs=0.03)
    assert s.mean() == pytest.approx(1.0, abs=2e-4)
    assert s.std() == pytest.approx(0.02, abs=2e-4)


def test_kitchen_uncertainty_reproducible_degenerate_and_validated():
    a = KitchenUncertainty().sample(50, 3)
    b = KitchenUncertainty().sample(50, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    T0, Tb, s = KitchenUncertainty(
        T_init_std_C=0.0, T_bath_shortfall_std_C=0.0, size_cv=0.0
    ).sample(5, 0)
    assert np.all(T0 == K0 + 20.0) and np.all(Tb == K0 + 100.0) and np.all(s == 1.0)
    for bad in ({"T_init_std_C": -1.0}, {"size_cv": -0.1}, {"size_cv": 0.2},
                {"T_bath_shortfall_std_C": -0.5}, {"T_init_mean_C": float("nan")}):
        with pytest.raises(ValueError):
            KitchenUncertainty(**bad)


# ---------------------------------------------------------------------------
# 2d. Corridor option
# ---------------------------------------------------------------------------

SPHERE_TIMES = np.arange(440.0, 801.0, 5.0)  # the sphere probe is in 63-70 C at 579-665 s


def test_corridor_price_is_notional_times_probability_with_binomial_se(sphere_response):
    n = 4000
    res = price_egg_corridor_option(SPHERE_TIMES, sphere_response, notional=7.0,
                                    n_paths=n, seed=5)
    p = res["prob_in_corridor"]
    assert np.allclose(res["prices"], 7.0 * p, rtol=0, atol=1e-12)
    assert np.allclose(res["price_se"], 7.0 * np.sqrt(p * (1 - p) / n), rtol=0, atol=1e-12)
    total = p + res["prob_undercooked"] + res["prob_overcooked"]
    assert np.allclose(total, 1.0, atol=1e-12)
    # undercooked early, overcooked late, corridor hit in between
    assert res["prob_undercooked"][0] > 0.99 and res["prob_overcooked"][-1] > 0.99
    assert np.all(np.diff(res["prob_undercooked"]) <= 1e-12)  # more cooking, less raw
    assert np.all(np.diff(res["prob_overcooked"]) >= -1e-12)
    # t_opt is the argmax of the price curve and the stored samples belong to it
    i = int(np.argmax(res["prices"]))
    assert res["t_opt_s"] == res["cook_times_s"][i]
    T_at_opt = res["T_samples_at_opt_C"]
    assert T_at_opt.shape == (n,)
    assert np.mean((T_at_opt >= 63.0) & (T_at_opt <= 70.0)) == pytest.approx(p[i], abs=1e-12)
    # price_at_opt comes from an independent path set: different value, same level
    assert res["price_at_opt"] != res["prices"][i]
    se = np.hypot(res["price_at_opt_se"], res["price_se"][i])
    assert abs(res["price_at_opt"] - res["prices"][i]) < 4.5 * se


def test_corridor_with_no_uncertainty_is_exactly_the_response_window(sphere_response):
    """Zero spread: the price is the notional exactly where the probe is in [63, 70] C."""
    unc = KitchenUncertainty(T_init_std_C=0.0, T_bath_shortfall_std_C=0.0, size_cv=0.0)
    times = np.arange(400.0, 900.0, 7.0)
    res = price_egg_corridor_option(times, sphere_response, uncertainty=unc, notional=10.0,
                                    n_paths=10, seed=1)
    T = sphere_response(times, K0 + 20.0, K0 + 100.0) - K0
    inside = (T >= 63.0) & (T <= 70.0)
    assert inside.sum() >= 5 and (~inside).sum() >= 5
    assert np.array_equal(res["prices"], np.where(inside, 10.0, 0.0))
    assert np.array_equal(res["prob_undercooked"], (T < 63.0).astype(float))
    assert np.array_equal(res["prob_overcooked"], (T > 70.0).astype(float))
    assert res["price_at_opt"] == 10.0 and res["price_at_opt_se"] == 0.0
    # custom window bounds are honored
    res2 = price_egg_corridor_option(times, sphere_response, K_low_C=66.0, K_high_C=68.0,
                                     uncertainty=unc, n_paths=10)
    T_in = (T >= 66.0) & (T <= 68.0)
    assert np.array_equal(res2["prices"], np.where(T_in, 10.0, 0.0))


def test_corridor_out_of_sample_price_removes_selection_bias(sphere_response):
    """The in-sample maximum is biased up; the price at t_opt on fresh paths is not."""
    times = np.arange(560.0, 700.0, 2.0)  # flat top: many near-ties for the argmax
    n, n_seeds = 300, 80
    # large-sample truth of the corridor probability at each t
    truth = price_egg_corridor_option(times, sphere_response, notional=1.0,
                                      n_paths=200000, seed=1234)["prob_in_corridor"]
    insample, oos = [], []
    for seed in range(n_seeds):
        r = price_egg_corridor_option(times, sphere_response, notional=1.0, n_paths=n, seed=seed)
        i = int(np.argmax(r["prices"]))
        insample.append(r["prices"][i] - truth[i])
        oos.append(r["price_at_opt"] - truth[i])
    insample, oos = np.array(insample), np.array(oos)
    sem_oos = oos.std(ddof=1) / np.sqrt(n_seeds)
    sem_in = insample.std(ddof=1) / np.sqrt(n_seeds)
    assert insample.mean() > 4.0 * sem_in           # clearly positive (selection bias)
    assert abs(oos.mean()) < 3.5 * sem_oos           # consistent with zero
    assert insample.mean() > oos.mean()


def test_corridor_larger_size_spread_lowers_the_max_price(sphere_response):
    base = price_egg_corridor_option(SPHERE_TIMES, sphere_response, n_paths=20000, seed=3)
    wide = price_egg_corridor_option(
        SPHERE_TIMES, sphere_response, uncertainty=KitchenUncertainty(size_cv=0.06),
        n_paths=20000, seed=3,
    )
    i, j = int(np.argmax(base["prices"])), int(np.argmax(wide["prices"]))
    gap = base["prices"][i] - wide["prices"][j]
    assert gap > 4.0 * np.hypot(base["price_se"][i], wide["price_se"][j])
    assert gap > 1.0  # and by a lot, in USD of a 10 USD notional


def test_corridor_is_reproducible_and_seed_dependent(sphere_response):
    a = price_egg_corridor_option(SPHERE_TIMES, sphere_response, n_paths=500, seed=11)
    b = price_egg_corridor_option(SPHERE_TIMES, sphere_response, n_paths=500, seed=11)
    c = price_egg_corridor_option(SPHERE_TIMES, sphere_response, n_paths=500, seed=12)
    assert np.array_equal(a["prices"], b["prices"]) and a["price_at_opt"] == b["price_at_opt"]
    assert not np.array_equal(a["prices"], c["prices"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"K_low_C": 70.0, "K_high_C": 63.0},
        {"K_low_C": 66.0, "K_high_C": 66.0},
        {"cook_times_s": []},
        {"cook_times_s": [300.0, 300.0, 400.0]},
        {"cook_times_s": [400.0, 300.0]},
        {"cook_times_s": [[300.0, 400.0]]},
        {"cook_times_s": [-5.0, 300.0]},
        {"n_paths": 1},
        {"notional": -1.0},
    ],
)
def test_corridor_rejects_invalid_inputs(sphere_response, kwargs):
    args = {"cook_times_s": [500.0, 600.0, 700.0], "n_paths": 100}
    args.update(kwargs)
    with pytest.raises(ValueError):
        price_egg_corridor_option(args.pop("cook_times_s"), sphere_response, **args)


def test_corridor_time_beyond_the_response_grid_is_an_error(sphere_response):
    with pytest.raises(ValueError, match="beyond the response grid"):
        price_egg_corridor_option([500.0, 1700.0], sphere_response, n_paths=50)


# ---------------------------------------------------------------------------
# 3. Culinary payoff and the American (optimal stopping) problem
# ---------------------------------------------------------------------------

def test_culinary_payoff_shape_and_defaults():
    # defaults: peak 100 at 66.5 C, zero at 63 and 70 C, symmetric
    assert float(egg_culinary_payoff(66.5)) == 100.0
    assert float(egg_culinary_payoff(63.0)) == 0.0 and float(egg_culinary_payoff(70.0)) == 0.0
    assert float(egg_culinary_payoff(50.0)) == 0.0 and float(egg_culinary_payoff(90.0)) == 0.0
    assert float(egg_culinary_payoff(65.0)) == pytest.approx(float(egg_culinary_payoff(68.0)))
    assert float(egg_culinary_payoff(65.0)) == pytest.approx(100.0 * (1 - (1.5 / 3.5) ** 2))
    assert float(egg_culinary_payoff(10.0, T_target_C=10.0, width_C=2.0, max_payoff=7.0)) == 7.0
    with pytest.raises(ValueError):
        egg_culinary_payoff(66.0, width_C=0.0)


def test_american_default_grid_covers_the_payoff_window(sphere_response, fem_response):
    # The grid starts before any kitchen reaches 63 C and ends after the slowest
    # kitchens leave the window (4 standard deviations of every parameter).
    for response in (sphere_response, fem_response):
        res = solve_american_egg_option(response, n_paths=200, n_paths_test=200, seed=0)
        grid = res["t_grid_s"]
        assert res["decision_interval_s"] == 5.0
        t63 = response.time_to_reach(63.0, T_ROOM, T_BOILING)
        t70 = response.time_to_reach(70.0, T_ROOM, T_BOILING)
        assert grid[0] < 0.8 * t63 and grid[-1] > 1.2 * t70
        assert grid[0] % 5.0 == 0.0 and grid[-1] % 5.0 == 0.0
        assert np.allclose(np.diff(grid), 5.0)
    # the default egg: soft window is 343-407 s
    res = solve_american_egg_option(fem_response, n_paths=200, n_paths_test=200, seed=0)
    assert 220.0 <= res["t_grid_s"][0] <= 260.0 and 560.0 <= res["t_grid_s"][-1] <= 600.0


def test_american_value_bounds_and_identities(sphere_response):
    res = solve_american_egg_option(sphere_response, n_paths=6000, n_paths_test=6000, seed=2)
    assert res["american_value"] <= res["perfect_foresight_value"]  # pathwise max bounds any rule
    assert res["best_fixed_value"] <= res["perfect_foresight_value"]
    assert res["value_of_information"] == pytest.approx(
        res["american_value"] - res["best_fixed_value"], abs=1e-12)
    assert res["american_value"] >= res["best_fixed_value"] - 3.0 * res["value_of_information_se"]
    assert res["value_of_information"] > 20.0  # an in-egg thermometer is worth a lot
    # the best fixed timer is read off the fixed-timer curve (time picked on the training set)
    grid, fixed = res["t_grid_s"], res["fixed_timer_values"]
    k = int(np.flatnonzero(grid == res["best_fixed_time_s"])[0])
    assert res["best_fixed_value"] == pytest.approx(fixed[k], abs=1e-9)
    assert res["best_fixed_value"] <= fixed.max() + 1e-12
    assert fixed.max() - res["best_fixed_value"] < 1.5  # picked a near-best time
    # stopping outputs: true temperatures at the stopping times, all within the grid
    assert res["stopping_times_s"].shape == res["stopping_temps_C"].shape == (6000,)
    assert grid[0] <= res["stopping_times_s"].min() and res["stopping_times_s"].max() <= grid[-1]
    assert 66.0 < np.mean(res["stopping_temps_C"]) < 67.0
    # the fitted rule generalizes: in-sample and out-of-sample values are close
    assert abs(res["american_value_in_sample"] - res["american_value"]) < 0.4


def test_american_is_deterministic_given_a_seed(sphere_response):
    a = solve_american_egg_option(sphere_response, n_paths=1500, n_paths_test=1500, seed=9)
    b = solve_american_egg_option(sphere_response, n_paths=1500, n_paths_test=1500, seed=9)
    assert a["american_value"] == b["american_value"]
    assert np.array_equal(a["stopping_times_s"], b["stopping_times_s"])


def test_american_rule_and_selections_use_the_training_paths_only(sphere_response):
    """Everything fitted or selected depends on (seed, n_paths), never on the test set."""
    a = solve_american_egg_option(sphere_response, n_paths=1500, n_paths_test=800, seed=3)
    b = solve_american_egg_option(sphere_response, n_paths=1500, n_paths_test=2500, seed=3)
    assert a["american_value_in_sample"] == b["american_value_in_sample"]
    assert np.array_equal(a["exercise_boundary_C"], b["exercise_boundary_C"], equal_nan=True)
    assert a["best_fixed_time_s"] == b["best_fixed_time_s"]
    assert a["threshold_rule"]["best_threshold_C"] == b["threshold_rule"]["best_threshold_C"]
    # but the evaluation does use the test set
    assert a["american_value"] != b["american_value"]
    assert a["stopping_times_s"].shape == (800,) and b["stopping_times_s"].shape == (2500,)


def test_american_best_fixed_timer_is_chosen_out_of_sample(sphere_response):
    """Selecting the timer on the same paths would make it equal the maximum of the
    fixed-timer curve; selecting on training paths gives a (slightly) lower test value."""
    gaps = []
    for seed in range(40):
        r = solve_american_egg_option(sphere_response, n_paths=300, n_paths_test=300, seed=seed)
        gaps.append(r["fixed_timer_values"].max() - r["best_fixed_value"])
    gaps = np.array(gaps)
    assert np.all(gaps >= -1e-12)
    sem = gaps.std(ddof=1) / np.sqrt(len(gaps))
    assert gaps.mean() > 2.5 * sem  # the in-sample maximum is biased up (measured: 0.37 +- 0.09)


def test_american_discounts_by_the_actual_gaps_of_a_non_uniform_grid(sphere_response):
    """No uncertainty, exact thermometer: the optimum is max_j exp(-r (t_j - t_0)) g(T_j)."""
    unc = KitchenUncertainty(T_init_std_C=0.0, T_bath_shortfall_std_C=0.0, size_cv=0.0)
    grid = np.array([500.0, 540.0, 560.0, 575.0, 585.0, 595.0, 605.0, 615.0, 635.0, 700.0])
    rate = 0.015  # 1/s: heavy enough that waiting costs more than a better temperature
    res = solve_american_egg_option(
        sphere_response, t_grid_s=grid, discount_rate=rate, obs_noise_C=0.0,
        uncertainty=unc, n_paths=200, n_paths_test=200, seed=0,
    )
    T = sphere_response(grid, K0 + 20.0, K0 + 100.0) - K0
    g = egg_culinary_payoff(T)
    discounted = np.exp(-rate * (grid - grid[0])) * g
    expected_idx = int(np.argmax(discounted))
    assert res["decision_interval_s"] is None
    assert res["american_value"] == pytest.approx(discounted[expected_idx], rel=1e-6)
    assert np.all(res["stopping_times_s"] == grid[expected_idx])
    assert res["american_value_se"] == pytest.approx(0.0, abs=1e-9)
    # a (wrong) constant-gap discount would have given a clearly different value
    wrong = np.exp(-rate * 5.0 * np.arange(len(grid))) * g
    assert abs(wrong.max() - discounted.max()) > 0.5
    # the undiscounted optimum is a later time: discounting changed the decision
    assert int(np.argmax(g)) > expected_idx
    # perfect foresight and the best fixed timer coincide with the optimum here
    assert res["perfect_foresight_value"] == pytest.approx(discounted.max(), rel=1e-9)
    assert res["best_fixed_value"] == pytest.approx(discounted.max(), rel=1e-9)


def test_american_value_falls_with_thermometer_noise(sphere_response):
    values = []
    for noise in (0.0, 0.5, 1.5):
        res = solve_american_egg_option(sphere_response, obs_noise_C=noise,
                                        n_paths=8000, n_paths_test=8000, seed=4)
        values.append((res["american_value"], res["american_value_se"]))
        assert res["american_value"] <= res["perfect_foresight_value"]
    (v0, s0), (v1, s1), (v2, s2) = values
    assert v0 > v1 + 3 * np.hypot(s0, s1)
    assert v1 > v2 + 3 * np.hypot(s1, s2)
    assert v0 > 99.0  # an exact thermometer comes within a grid step of the ceiling


def test_american_exercise_boundary_and_threshold_benchmark(sphere_response):
    res = solve_american_egg_option(sphere_response, n_paths=8000, n_paths_test=8000, seed=6)
    boundary = res["exercise_boundary_C"]
    assert boundary.shape == res["t_grid_s"].shape
    assert np.isnan(boundary[-1]) and np.isnan(boundary[0])  # forced / never exercised
    finite = boundary[np.isfinite(boundary)]
    assert len(finite) > 20
    assert 65.0 < np.median(finite) < 67.0  # close to, a bit below, the 66.5 C target
    assert np.all((finite >= 66.5 - 5.0) & (finite <= 66.5 + 5.0))  # inside the regression window
    thr = res["threshold_rule"]
    assert thr["values"].shape == thr["thresholds_C"].shape == thr["values_se"].shape
    assert thr["best_value"] <= res["perfect_foresight_value"]
    assert thr["best_threshold_C"] == pytest.approx(66.5, abs=1.0)
    # the values rise to an interior optimum and fall again
    k = int(np.argmax(thr["values"]))
    assert 0 < k < len(thr["values"]) - 1
    # the fitted rule and the one-parameter threshold rule are close: a few tenths of a point
    # of 100, in either direction (the sign depends on the response and the noise)
    assert abs(res["american_value"] - thr["best_value"]) < 1.0


def test_threshold_benchmark_default_grid_is_fine_and_not_worse_than_a_coarse_grid(sphere_response):
    """On a 0.5 C grid the one-parameter rule loses up to 0.2 points of 100 against its
    optimum, which would flatter the fitted rule in any comparison with it."""
    kwargs = dict(n_paths=8000, n_paths_test=8000, seed=6)
    res = solve_american_egg_option(sphere_response, **kwargs)
    grid = res["threshold_rule"]["thresholds_C"]
    assert grid[0] == pytest.approx(63.0) and grid[-1] == pytest.approx(70.0)
    assert np.allclose(np.diff(grid), 0.02)
    coarse = solve_american_egg_option(
        sphere_response, thresholds_C=np.arange(63.0, 70.01, 0.5), **kwargs
    )
    # same paths and selection rule; the fine grid contains every coarse threshold, so its
    # test value is not lower beyond Monte Carlo noise
    se = res["threshold_rule"]["best_value_se"]
    assert res["threshold_rule"]["best_value"] >= coarse["threshold_rule"]["best_value"] - 2 * se
    # the grid is passed through unchanged when given
    assert np.array_equal(coarse["threshold_rule"]["thresholds_C"], np.arange(63.0, 70.01, 0.5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_grid_s": [400.0]},
        {"t_grid_s": [400.0, 400.0, 500.0]},
        {"t_grid_s": [500.0, 400.0]},
        {"t_grid_s": [[400.0, 500.0]]},
        {"obs_noise_C": -0.1},
        {"discount_rate": -0.01},
        {"width_C": 0.0},
        {"max_payoff": 0.0},
        {"n_paths": 1},
        {"n_paths_test": 1},
        {"thresholds_C": []},
        {"poly_degree": -1},
    ],
)
def test_american_rejects_invalid_inputs(sphere_response, kwargs):
    args = {"n_paths": 100, "n_paths_test": 100}
    args.update(kwargs)
    with pytest.raises(ValueError):
        solve_american_egg_option(sphere_response, **args)


# ---------------------------------------------------------------------------
# Headline numbers of the shipped configuration (FEM white_min response)
# ---------------------------------------------------------------------------

def test_headline_numbers_for_the_default_egg(fem_response):
    """Pinned, seeded results; the tolerances are a few standard errors wide."""
    times = np.arange(250.0, 601.0, 2.0)
    cor = price_egg_corridor_option(times, fem_response, n_paths=20000, seed=42)
    i = int(np.argmax(cor["prices"]))
    assert cor["prices"][i] == pytest.approx(8.87, abs=0.12)
    assert cor["price_se"][i] == pytest.approx(0.0224, abs=0.002)
    assert cor["t_opt_s"] == pytest.approx(378.0, abs=6.0)
    assert cor["price_at_opt"] == pytest.approx(8.83, abs=0.12)
    assert cor["prob_in_corridor"][i] == pytest.approx(0.887, abs=0.012)

    am = solve_american_egg_option(fem_response)  # shipped defaults: 20000 + 20000 paths
    assert am["american_value"] == pytest.approx(98.14, abs=0.15)
    assert am["best_fixed_value"] == pytest.approx(67.1, abs=1.0)
    assert am["best_fixed_time_s"] == pytest.approx(380.0, abs=10.0)
    assert am["perfect_foresight_value"] == pytest.approx(99.80, abs=0.05)
    assert am["value_of_information"] == pytest.approx(31.0, abs=1.2)
    assert am["threshold_rule"]["best_value"] == pytest.approx(98.32, abs=0.15)
    assert am["threshold_rule"]["best_threshold_C"] == pytest.approx(66.3, abs=0.3)
    assert am["american_value"] >= am["best_fixed_value"] - 3.0 * am["value_of_information_se"]


def test_public_exports():
    import egg_fem

    for name in ("ThermalResponse", "KitchenUncertainty", "check_similarity_against_fem",
                 "black_scholes_call", "solve_black_scholes_heat_fem",
                 "fourier_sphere_center_temp", "price_egg_corridor_option",
                 "egg_culinary_payoff", "solve_american_egg_option"):
        assert name in egg_fem.__all__ and hasattr(egg_fem, name)
