"""Tests for the FEM core of egg_fem: materials, geometry, solver and doneness.

Pinned numbers (cooking times, volumes) were computed with the shipped code
on the default egg (55 x 42 mm, 15 mm yolk, 1.5 mm mesh, dt = 1 s,
Dirichlet 100 °C shell, 20 °C start) and are guarded with small tolerances.
Tolerances on discretization-sensitive quantities are the measured values
plus a margin, not arbitrary round numbers.
"""

import glob
import os
import tempfile
import threading
import inspect

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.sparse import csc_matrix
from skfem import MeshTri

import gmsh

from egg_fem import (
    H_BOILING_SHELL,
    T_BOILING,
    T_FRIDGE,
    T_ROOM,
    WHITE_CONST,
    WHITE_DENATURATION_K,
    YOLK_CONST,
    YOLK_FULL_SET_K,
    HeatSolver,
    center_crossing_times,
    create_circular_mesh,
    create_egg_mesh,
    crossing_time,
    doneness_times,
    egg_profile,
    egg_white_denaturation_properties,
    egg_white_properties,
    egg_yolk_denaturation_properties,
    egg_yolk_properties,
    mesh_volume,
    solve_condense,
)

EGG_MATERIALS = {"white": egg_white_properties, "yolk": egg_yolk_properties}
EGG_MATERIALS_DENAT = {
    "white": egg_white_denaturation_properties,
    "yolk": egg_yolk_denaturation_properties,
}


def _profile_volume(n=4001, **kwargs):
    """Volume of the solid of revolution of egg_profile, by the trapezoid rule."""
    r, z = egg_profile(n=n, **kwargs)
    return float(np.sum(0.5 * (r[1:] ** 2 + r[:-1] ** 2) * np.abs(np.diff(z))) * np.pi)


# ---------------------------------------------------------------------------
# Shared fixtures (module scope: each expensive run happens once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_egg_mesh():
    return create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002)


@pytest.fixture(scope="module")
def default_egg_mesh():
    return create_egg_mesh()


@pytest.fixture(scope="module")
def default_egg_run(default_egg_mesh):
    """Default egg from 20 °C into a 100 °C Dirichlet bath, dt = 1 s, to 800 s."""
    solver = HeatSolver(
        default_egg_mesh, EGG_MATERIALS, T_init=T_ROOM, T_boundary=T_BOILING, dt=1.0
    )
    solver.run(800.0, progress=False)
    return solver


# ---------------------------------------------------------------------------
# 1. Material properties
# ---------------------------------------------------------------------------

def test_material_properties_bounds():
    """Material properties stay within the measured ranges over [0 °C, 100 °C]."""
    T = np.linspace(273.15, 373.15, 50)

    for mat in (egg_white_properties, egg_yolk_properties, WHITE_CONST, YOLK_CONST):
        rho = mat.density(T)
        kappa = mat.conductivity(T)
        cp = mat.specific_heat(T)
        rho_cp = mat.rho_cp(T)

        assert np.all(rho > 1000.0) and np.all(rho < 1060.0)
        assert np.all(kappa > 0.3) and np.all(kappa < 0.62)
        assert np.all(cp > 3200.0) and np.all(cp < 3950.0)
        assert np.allclose(rho_cp, rho * cp)


def test_material_fits_at_reference_temperature():
    """The linear fits are anchored at 20 °C (guards against accidental edits)."""
    T = 293.15
    assert egg_white_properties.density(T) == pytest.approx(1040.0)
    assert egg_white_properties.conductivity(T) == pytest.approx(0.50)
    assert egg_white_properties.specific_heat(T) == pytest.approx(3800.0)
    assert egg_yolk_properties.density(T) == pytest.approx(1030.0)
    assert egg_yolk_properties.conductivity(T) == pytest.approx(0.32)
    assert egg_yolk_properties.specific_heat(T) == pytest.approx(3300.0)


def test_latent_heat_denaturation():
    """Effective specific heat integrates to base heat + latent heat."""
    mat_base = egg_white_properties
    mat_denat = egg_white_denaturation_properties

    # Peak occurs at denaturation_temp
    T_d = mat_denat.denaturation_temp
    assert mat_denat.effective_specific_heat(T_d) > mat_base.effective_specific_heat(T_d)

    # Integral of (Cp_eff - Cp_base) over a wide temperature range equals latent_heat
    excess_cp = lambda T: mat_denat.effective_specific_heat(T) - mat_base.effective_specific_heat(T)
    integral, _ = quad(excess_cp, T_d - 30.0, T_d + 30.0)
    assert np.isclose(integral, mat_denat.latent_heat, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Geometry and meshes
# ---------------------------------------------------------------------------

def test_egg_profile_dimensions_and_shape():
    """Exact length and breadth, monotone z, pointed top, blunt-side maximum."""
    L, B, asym = 0.055, 0.042, 0.06
    r, z = egg_profile(n=4001, length=L, breadth=B, asymmetry=asym)
    assert r.shape == (4001,) and z.shape == (4001,)

    # Exact length (z is analytic at the poles) and exact poles on the axis
    assert z[0] - z[-1] == pytest.approx(L, abs=1e-15)
    assert r[0] == 0.0 and r[-1] == 0.0
    assert np.all(r >= 0.0)
    # Strictly monotone z: no dimples at the poles
    assert np.all(np.diff(z) < 0.0)

    # Breadth: the analytic maximum radius is B/2 at z = -asymmetry * L
    assert np.max(r) == pytest.approx(B / 2, rel=1e-6)
    assert z[np.argmax(r)] == pytest.approx(-asym * L, abs=1e-5)

    # Pointed top (+z), blunt bottom (-z)
    z_probe = 0.8 * L / 2
    r_top = np.interp(z_probe, z[::-1], r[::-1])
    r_bottom = np.interp(-z_probe, z[::-1], r[::-1])
    assert r_top < r_bottom
    # Reference values for the default egg quoted in the docstring
    assert np.interp(0.020, z[::-1], r[::-1]) == pytest.approx(0.0132, abs=1e-4)
    assert np.interp(-0.020, z[::-1], r[::-1]) == pytest.approx(0.0157, abs=1e-4)


def test_egg_profile_default_resolution_and_volume():
    """The default profile (n = 200) reaches the breadth to 1e-5; volume about 50.65 mL."""
    r, z = egg_profile()
    assert r.shape == (200,)
    assert np.max(r) == pytest.approx(0.021, rel=1e-5)
    vol = _profile_volume()
    assert vol == pytest.approx(50.65e-6, abs=0.02e-6)
    # Shape coefficient V / (L B^2) of a hen egg is about 0.5
    assert vol / (0.055 * 0.042**2) == pytest.approx(0.522, abs=0.002)


def test_egg_profile_zero_asymmetry_is_ellipse():
    r, z = egg_profile(n=101, length=0.05, breadth=0.04, asymmetry=0.0)
    expected = 0.02 * np.sqrt(np.clip(1.0 - (2.0 * z / 0.05) ** 2, 0.0, None))
    assert np.allclose(r, expected, atol=1e-12)
    # Mirror symmetric about z = 0
    assert np.allclose(r, r[::-1], atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(length=-0.05),
        dict(breadth=0.0),
        dict(asymmetry=-0.01),
        dict(asymmetry=0.25),
        dict(n=2),
    ],
)
def test_egg_profile_validation(kwargs):
    with pytest.raises(ValueError):
        egg_profile(**kwargs)


def test_circular_mesh_and_tempfile_cleanup():
    """Circular mesh creation leaves no temporary .msh files behind."""
    tmp_dir = tempfile.gettempdir()
    before_files = set(glob.glob(os.path.join(tmp_dir, "*.msh")))

    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)

    after_files = set(glob.glob(os.path.join(tmp_dir, "*.msh")))
    assert after_files == before_files, "Temporary .msh file was not cleaned up!"

    assert "shell" in mesh.boundaries
    assert "axis" in mesh.boundaries
    assert "egg" in mesh.subdomains
    assert mesh.p.shape[1] > 10


def test_mesh_builders_print_nothing(capfd):
    """meshio.read prints a stray blank line; the library keeps stdout and stderr clean."""
    create_circular_mesh(radius=0.02, mesh_size=0.005)
    create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002)
    captured = capfd.readouterr()
    assert captured.out == "" and captured.err == ""


def test_circular_mesh_volume_is_sphere():
    """mesh_volume of the semicircle is the sphere volume (polygonal approximation, below it)."""
    R = 0.021
    mesh = create_circular_mesh(radius=R, mesh_size=0.0015)
    sphere = 4.0 / 3.0 * np.pi * R**3
    assert mesh_volume(mesh) == pytest.approx(sphere, rel=3e-3)
    assert mesh_volume(mesh) < sphere
    assert mesh_volume(mesh, "egg") == pytest.approx(mesh_volume(mesh), rel=1e-12)


def test_egg_mesh_subdomains(coarse_egg_mesh):
    """The egg mesh contains yolk and white subdomains and the named boundaries."""
    mesh = coarse_egg_mesh
    assert "shell" in mesh.boundaries
    assert "axis" in mesh.boundaries
    assert "white" in mesh.subdomains
    assert "yolk" in mesh.subdomains
    assert len(mesh.subdomains["white"]) > 0
    assert len(mesh.subdomains["yolk"]) > 0
    # The two subdomains partition the mesh
    n_elem = mesh.t.shape[1]
    both = np.concatenate([mesh.subdomains["white"], mesh.subdomains["yolk"]])
    assert len(both) == n_elem and len(np.unique(both)) == n_elem


def test_egg_mesh_volumes(default_egg_mesh):
    """Mesh volume matches the analytic profile; the yolk is about 28% of the egg."""
    mesh = default_egg_mesh
    vol = mesh_volume(mesh)
    vol_yolk = mesh_volume(mesh, "yolk")
    vol_white = mesh_volume(mesh, "white")
    assert vol_yolk + vol_white == pytest.approx(vol, rel=1e-12)

    # Mesh (polygonal) volume is just below the analytic profile volume.
    # Measured: 50.646 mL vs 50.653 mL (1.4e-4 relative).
    analytic = _profile_volume()
    assert vol < analytic
    assert vol == pytest.approx(analytic, rel=5e-4)
    assert vol == pytest.approx(50.65e-6, abs=0.02e-6)

    # Yolk: sphere of radius 15 mm, 14.14 mL; mesh 14.10 mL (0.25% below)
    sphere = 4.0 / 3.0 * np.pi * 0.015**3
    assert vol_yolk == pytest.approx(sphere, rel=6e-3)
    assert vol_yolk / vol == pytest.approx(0.2785, abs=0.003)
    # Real yolks are about 26-31% of the egg
    assert 0.26 < vol_yolk / vol < 0.31


def test_egg_mesh_outline_dimensions(coarse_egg_mesh):
    """Mesh extents match length and breadth; the maximum radius is on the blunt side."""
    p = coarse_egg_mesh.p
    assert np.max(p[1]) - np.min(p[1]) == pytest.approx(0.055, abs=1e-9)
    assert np.max(p[0]) == pytest.approx(0.021, rel=2e-3)
    assert np.min(p[0]) == pytest.approx(0.0, abs=1e-12)
    shell_nodes = np.unique(coarse_egg_mesh.facets[:, coarse_egg_mesh.boundaries["shell"]])
    assert p[1, shell_nodes[np.argmax(p[0, shell_nodes])]] < 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(yolk_radius=0.025),                       # larger than the egg half-breadth
        dict(yolk_radius=0.015, yolk_center_z=0.014),  # sticks out of the pointed top
        dict(yolk_radius=0.015, yolk_center_z=-0.015),  # sticks out of the blunt bottom
        dict(yolk_radius=0.0215),                      # touches/crosses the equator
    ],
)
def test_egg_mesh_rejects_yolk_that_does_not_fit(kwargs):
    with pytest.raises(ValueError, match="does not fit"):
        create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002, **kwargs)


def test_egg_mesh_rejects_bad_sizes():
    with pytest.raises(ValueError):
        create_egg_mesh(yolk_radius=0.0)
    with pytest.raises(ValueError):
        create_egg_mesh(mesh_size=-1.0)
    with pytest.raises(ValueError):
        create_egg_mesh(asymmetry=0.3)


def test_mesh_volume_unknown_subdomain(coarse_egg_mesh):
    with pytest.raises(ValueError, match="subdomain"):
        mesh_volume(coarse_egg_mesh, "albumen")


def test_gmsh_standalone_builders_finalize():
    """Without a caller session the builders initialize and finalize gmsh themselves."""
    assert not gmsh.isInitialized()
    create_circular_mesh(radius=0.02, mesh_size=0.005)
    assert not gmsh.isInitialized()
    create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002)
    assert not gmsh.isInitialized()


def test_gmsh_builders_preserve_caller_session():
    """An already-initialized gmsh session, its model and options survive the builders."""
    assert not gmsh.isInitialized()
    gmsh.initialize()
    try:
        gmsh.model.add("caller_model")
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 1.0)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("General.Verbosity", 3)
        models_before = list(gmsh.model.list())
        entities_before = gmsh.model.getEntities()

        mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
        egg = create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002)

        assert mesh.p.shape[1] > 10 and egg.p.shape[1] > 10
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "caller_model"
        assert list(gmsh.model.list()) == models_before
        assert gmsh.model.getEntities() == entities_before
        assert gmsh.option.getNumber("General.Verbosity") == 3
    finally:
        gmsh.finalize()


def test_gmsh_session_cleaned_up_after_failure(monkeypatch):
    """If mesh generation fails, the caller's session is still restored."""
    gmsh.initialize()
    try:
        gmsh.model.add("caller_model")
        models_before = list(gmsh.model.list())

        def boom():
            raise RuntimeError("mesh export failed")

        monkeypatch.setattr("egg_fem.geometry._gmsh_to_skfem", boom)
        with pytest.raises(RuntimeError, match="mesh export failed"):
            create_circular_mesh(radius=0.02, mesh_size=0.005)
        assert gmsh.model.getCurrent() == "caller_model"
        assert list(gmsh.model.list()) == models_before
    finally:
        gmsh.finalize()


@pytest.mark.skipif(
    "interruptible" not in inspect.signature(gmsh.initialize).parameters,
    reason="off-main-thread gmsh needs gmsh >= 4.12",
)
def test_mesh_creation_off_main_thread():
    """With gmsh >= 4.12 the builders work from a worker thread."""
    result = {}

    def work():
        try:
            result["mesh"] = create_circular_mesh(radius=0.02, mesh_size=0.005)
        except Exception as exc:  # pragma: no cover - reported below
            result["error"] = exc

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert "error" not in result, result.get("error")
    assert result["mesh"].p.shape[1] > 10


# ---------------------------------------------------------------------------
# 3. Analytic benchmark: sphere center temperature
# ---------------------------------------------------------------------------

def _sphere_center_theta(t, R, alpha, n_terms=400):
    """Exact (T - T_bath) / (T_init - T_bath) at the center of a sphere.

    Sphere of radius R, uniform initial temperature, surface held at the bath
    temperature: theta = 2 * sum_n (-1)^(n+1) exp(-n^2 pi^2 alpha t / R^2).
    """
    n = np.arange(1, n_terms + 1)
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    return 2.0 * np.sum(
        (-1.0) ** (n + 1) * np.exp(-((n * np.pi) ** 2) * alpha * t / R**2), axis=1
    )


@pytest.fixture(scope="module")
def sphere_benchmark():
    R = 0.021
    mesh = create_circular_mesh(radius=R, mesh_size=0.0015)
    solver = HeatSolver(mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=1.0)
    solver.run(720.0, progress=False)
    alpha = WHITE_CONST.conductivity(0.0) / (
        WHITE_CONST.density(0.0) * WHITE_CONST.specific_heat(0.0)
    )
    return solver, R, alpha


def test_dirichlet_matches_exact_sphere_center_temperature(sphere_benchmark):
    """Center temperature of the axisymmetric FEM vs the exact sphere series.

    Constant properties, Dirichlet shell.  Measured maximum deviation over
    60-720 s is 0.078 K on a 20-100 K range (backward Euler with dt = 1 s and
    1.5 mm elements); a planar (r-weight 1) formulation misses by several K.
    """
    solver, R, alpha = sphere_benchmark
    times, T_center = solver.center_temperature_history()
    window = (times >= 60.0) & (times <= 720.0)
    exact = (T_BOILING - 273.15) + (T_ROOM - T_BOILING) * _sphere_center_theta(
        times[window], R, alpha
    )
    assert np.max(np.abs(T_center[window] - exact)) < 0.12
    # The curve is far from trivial over this window: it spans most of the range
    assert exact[0] < 20.1 and exact[-1] > 80.0


def test_sphere_center_crossing_time(sphere_benchmark):
    """Time for the sphere center to reach 63 °C vs the exact series (exact: 465.07 s)."""
    solver, R, alpha = sphere_benchmark
    target = (63.0 + 273.15 - T_BOILING) / (T_ROOM - T_BOILING)
    t_exact = brentq(lambda t: _sphere_center_theta(t, R, alpha)[0] - target, 300.0, 700.0)
    assert t_exact == pytest.approx(465.07, abs=0.05)

    times, T_center = solver.center_temperature_history()
    t_fem = crossing_time(times, T_center, 63.0)
    # Measured offset: -0.39 s
    assert abs(t_fem - t_exact) < 1.0


# ---------------------------------------------------------------------------
# 4. Solver: boundary conditions and physics
# ---------------------------------------------------------------------------

def test_dirichlet_solver_bounded_and_monotonic():
    """Dirichlet solver: bounded by the BCs up to a small undershoot, center heats monotonically."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.003)
    solver = HeatSolver(mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0)
    times, temps = solver.run(t_end=600.0, progress=False)

    assert len(times) == 121
    all_T = np.array(temps)
    # Upper bound is exact (discrete maximum principle holds from above here)
    assert np.all(all_T <= T_BOILING + 1e-9)
    # Consistent-mass undershoot below T_init in the first steps: measured 1.68 K at dt = 5 s
    assert np.min(all_T) >= T_ROOM - 2.0
    assert np.min(all_T) < T_ROOM - 0.5  # the undershoot is real; documented in HeatSolver
    for T in temps:
        assert np.allclose(T[solver.shell_dofs], T_BOILING)

    _, T_center = solver.center_temperature_history()
    # The center increases monotonically from step 13 on (measured), 150 s here
    assert np.all(np.diff(T_center[30:]) > 0.0)
    assert T_center[-1] == pytest.approx(79.85, abs=0.2)


def test_dirichlet_steady_state_is_uniform():
    """After many diffusion times the whole field equals the shell temperature."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.004)
    solver = HeatSolver(mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=50.0)
    solver.run(t_end=5000.0, progress=False)
    assert np.allclose(solver.T, T_BOILING, atol=1e-6)


def test_dirichlet_initial_state_has_shell_at_boundary_temperature():
    """Documented convention: Dirichlet starts with the shell at T_boundary, Robin uniform."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    dirichlet = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)
    robin = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0, h_conv=H_BOILING_SHELL)

    assert np.all(dirichlet.temperatures[0][dirichlet.shell_dofs] == T_BOILING)
    assert np.all(dirichlet.temperatures[0][dirichlet.interior_dofs] == T_ROOM)
    assert np.all(robin.temperatures[0] == T_ROOM)
    assert np.array_equal(dirichlet.T, dirichlet.temperatures[0])


def test_robin_boundary_condition():
    """Robin BC: insulated shell keeps T_init; finite h lags the Dirichlet solution."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.004)
    kwargs = dict(T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0)

    solver_dirichlet = HeatSolver(mesh, WHITE_CONST, h_conv=None, **kwargs)
    solver_robin = HeatSolver(mesh, WHITE_CONST, h_conv=H_BOILING_SHELL, **kwargs)
    solver_insulated = HeatSolver(mesh, WHITE_CONST, h_conv=0.0, **kwargs)
    for solver in (solver_dirichlet, solver_robin, solver_insulated):
        solver.run(t_end=300.0, progress=False)

    # Insulated boundary (h_conv = 0) keeps the temperature uniform at T_ROOM
    assert np.allclose(solver_insulated.T, T_ROOM, atol=1e-9)

    # Robin shell heats towards T_BOILING but stays below it; with h = 2000
    # the shell is within about 1 K of the bath at 300 s (measured 98.97-99.15 °C)
    T_shell = solver_robin.T[solver_robin.shell_dofs] - 273.15
    assert np.all(T_shell > 98.5) and np.all(T_shell < 99.5)

    # Finite h lags infinite h (Dirichlet): measured 42.88 vs 46.05 °C at 300 s
    _, Tc_dirichlet = solver_dirichlet.center_temperature_history()
    _, Tc_robin = solver_robin.center_temperature_history()
    assert Tc_robin[-1] == pytest.approx(42.88, abs=0.2)
    assert Tc_dirichlet[-1] == pytest.approx(46.05, abs=0.2)
    assert Tc_robin[-1] < Tc_dirichlet[-1]


def test_robin_converges_to_dirichlet_for_large_h():
    """As h_conv -> infinity, the Robin solution converges to the Dirichlet solution."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.004)

    solver_dir = HeatSolver(mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=2.0)
    solver_high_h = HeatSolver(
        mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=2.0, h_conv=1e8
    )
    # By convention a Robin run starts uniform while a Dirichlet run starts with
    # the shell already at T_boundary (a mesh-size-dependent offset of about 2 K
    # here).  Give the Robin run the Dirichlet initial state so the limit
    # itself is tested.
    solver_high_h.T[solver_high_h.shell_dofs] = T_BOILING
    solver_high_h.temperatures[0] = solver_high_h.T.copy()

    solver_dir.run(t_end=200.0, progress=False)
    solver_high_h.run(t_end=200.0, progress=False)

    _, Tc_dir = solver_dir.center_temperature_history()
    _, Tc_high_h = solver_high_h.center_temperature_history()
    assert np.allclose(Tc_dir, Tc_high_h, atol=1e-3)
    assert np.allclose(solver_dir.T, solver_high_h.T, atol=1e-3)


def test_set_boundary_conditions_keeps_bc_type_when_h_omitted():
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)

    robin = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0, h_conv=H_BOILING_SHELL)
    robin.set_boundary_conditions(300.0)
    assert robin.h_conv == H_BOILING_SHELL and robin.T_boundary == 300.0
    robin.set_boundary_conditions(310.0, h_conv=500.0)
    assert robin.h_conv == 500.0 and robin.T_boundary == 310.0
    robin.set_boundary_conditions(320.0, h_conv=None)  # None -> Dirichlet
    assert robin.h_conv is None and robin.T_boundary == 320.0

    dirichlet = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)
    dirichlet.set_boundary_conditions(280.0)
    assert dirichlet.h_conv is None and dirichlet.T_boundary == 280.0
    dirichlet.set_boundary_conditions(280.0, h_conv=0.0)  # insulated Robin
    assert dirichlet.h_conv == 0.0

    with pytest.raises(ValueError):
        dirichlet.set_boundary_conditions(280.0, h_conv=-1.0)


def test_set_boundary_conditions_does_not_modify_state():
    """Mid-run BC changes never desynchronize T from the stored history."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)
    solver.run(50.0, progress=False)
    n = len(solver.temperatures)
    before = solver.T.copy()

    for args in ((273.15, H_BOILING_SHELL), (300.0,), (273.15, None), (350.0, 0.0)):
        solver.set_boundary_conditions(*args)
        assert np.array_equal(solver.T, solver.temperatures[-1])
        assert np.array_equal(solver.T, before)
        assert len(solver.temperatures) == n and len(solver.times) == n


def test_boundary_condition_switch_boil_then_ice_bath(coarse_egg_mesh):
    """Boil 600 s, then an ice bath (Robin, 0 °C): the center peaks, then cools."""
    solver = HeatSolver(coarse_egg_mesh, EGG_MATERIALS, T_ROOM, T_BOILING, dt=5.0)
    solver.run(600.0, progress=False)
    n_boil = len(solver.times)
    _, Tc_boil = solver.center_temperature_history()

    solver.set_boundary_conditions(273.15, H_BOILING_SHELL)
    assert np.array_equal(solver.T, solver.temperatures[-1])
    solver.run(1800.0, progress=False)

    assert np.array_equal(solver.T, solver.temperatures[-1])
    assert len(solver.times) == len(solver.temperatures) == 361
    assert np.all(np.diff(solver.times) == pytest.approx(5.0))

    times, Tc = solver.center_temperature_history()
    i_peak = int(np.argmax(Tc))
    # The center keeps heating for a while after the switch (measured: 61.0 °C
    # at 600 s, peak 76.1 °C at 820 s), then cools towards the bath (12.5 °C at 1800 s)
    assert i_peak > n_boil
    assert Tc[i_peak] > Tc_boil[-1] + 10.0
    assert times[i_peak] == pytest.approx(820.0, abs=40.0)
    assert Tc[-1] < Tc[i_peak] - 50.0
    assert Tc[-1] > 0.0
    # The shell follows the ice bath
    assert np.all(solver.T[solver.shell_dofs] < 20.0 + 273.15)


def test_dirichlet_ice_bath_switch_imposes_new_shell_temperature():
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)
    solver.run(100.0, progress=False)
    solver.set_boundary_conditions(273.15)  # still Dirichlet
    solver.run(105.0, progress=False)
    assert np.allclose(solver.T[solver.shell_dofs], 273.15)
    assert np.array_equal(solver.T, solver.temperatures[-1])


def test_latent_heat_slows_heating(coarse_egg_mesh):
    """Endothermic latent heat of denaturation slows center heating slightly."""
    kwargs = dict(T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0)
    solver_std = HeatSolver(coarse_egg_mesh, EGG_MATERIALS, **kwargs)
    solver_denat = HeatSolver(coarse_egg_mesh, EGG_MATERIALS_DENAT, **kwargs)
    solver_std.run(t_end=600.0, progress=False)
    solver_denat.run(t_end=600.0, progress=False)

    _, Tc_std = solver_std.center_temperature_history()
    _, Tc_denat = solver_denat.center_temperature_history()

    # Measured at 600 s: 60.99 °C without and 60.37 °C with latent heat
    diff = Tc_std[-1] - Tc_denat[-1]
    assert 0.3 < diff < 1.2


# ---------------------------------------------------------------------------
# 5. Solver: validation, stepping, probes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_dt_raises(dt):
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    with pytest.raises(ValueError, match="dt"):
        HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=dt)


@pytest.mark.parametrize("h", [-1.0, float("nan")])
def test_invalid_h_conv_raises(h):
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    with pytest.raises(ValueError, match="h_conv"):
        HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=1.0, h_conv=h)


def test_material_dict_must_match_mesh_subdomains(coarse_egg_mesh):
    # A key that is not a subdomain of the mesh
    with pytest.raises(ValueError, match="albumen"):
        HeatSolver(
            coarse_egg_mesh,
            {"yolk": egg_yolk_properties, "albumen": egg_white_properties},
            T_ROOM, T_BOILING, dt=5.0,
        )
    # A listed subdomain does not exist on a single-region mesh
    circle = create_circular_mesh(radius=0.02, mesh_size=0.005)
    with pytest.raises(ValueError, match="not subdomains of the mesh"):
        HeatSolver(
            circle, {"white": egg_white_properties, "yolk": egg_yolk_properties},
            T_ROOM, T_BOILING, dt=5.0,
        )
    # Elements without a material: no silent default to the white
    with pytest.raises(ValueError, match="not in any of the subdomains"):
        HeatSolver(coarse_egg_mesh, {"white": egg_white_properties}, T_ROOM, T_BOILING, dt=5.0)
    with pytest.raises(ValueError, match="not in any of the subdomains"):
        HeatSolver(coarse_egg_mesh, {"yolk": egg_yolk_properties}, T_ROOM, T_BOILING, dt=5.0)


def test_material_dict_rejects_overlapping_subdomains():
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    everything = np.arange(mesh.t.shape[1])
    mesh = mesh.with_subdomains({"a": everything, "b": everything[:5]})
    with pytest.raises(ValueError, match="more than one"):
        HeatSolver(mesh, {"a": WHITE_CONST, "b": YOLK_CONST}, T_ROOM, T_BOILING, dt=5.0)


def test_material_dict_without_subdomains_raises():
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005).with_subdomains({})
    with pytest.raises(ValueError, match="subdomains"):
        HeatSolver(mesh, {"white": WHITE_CONST}, T_ROOM, T_BOILING, dt=5.0)


def test_multi_region_with_identical_materials_equals_single_region(coarse_egg_mesh):
    """A dict with the same material in both regions reproduces the single-material run."""
    kwargs = dict(T_init=T_ROOM, T_boundary=T_BOILING, dt=10.0)
    single = HeatSolver(coarse_egg_mesh, WHITE_CONST, **kwargs)
    multi = HeatSolver(coarse_egg_mesh, {"white": WHITE_CONST, "yolk": WHITE_CONST}, **kwargs)
    single.run(100.0, progress=False)
    multi.run(100.0, progress=False)
    assert np.allclose(single.T, multi.T, atol=1e-9)


def test_run_step_count_is_float_safe():
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)

    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=1.0)
    solver.run(10.0, progress=False)
    assert len(solver.times) == 11 and solver.time == pytest.approx(10.0)
    solver.run(20.0, progress=False)
    assert len(solver.times) == 21 and solver.time == pytest.approx(20.0)
    solver.run(20.0, progress=False)  # nothing left to do
    solver.run(5.0, progress=False)
    assert len(solver.times) == 21

    # dt = 0.1 is not exactly representable: accumulated time must not cause an extra step
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=0.1)
    solver.run(6.0, progress=False)
    solver.run(6.0, progress=False)
    solver.run(12.0, progress=False)
    assert len(solver.times) == 121 and len(solver.temperatures) == 121

    # t_end not a multiple of dt: whole steps until time >= t_end
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=0.3)
    solver.run(10.0, progress=False)
    assert len(solver.times) == 35
    assert solver.time >= 10.0 and solver.time - 10.0 < 0.3


@pytest.mark.parametrize("dt, expected_times", [(1.0, [60, 120]), (7.0, [63, 126]), (90.0, [90, 180])])
def test_progress_reported_every_60_simulated_seconds(capsys, dt, expected_times):
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=dt)
    # With dt = 90 s, run(130) takes two steps (to 180 s): one line per step at most
    solver.run(130.0, progress=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "T_center" in ln]
    reported = [float(ln.split("t =")[1].split("s")[0]) for ln in lines]
    assert reported == pytest.approx(expected_times)

    # No output when progress is off
    solver2 = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=dt)
    solver2.run(130.0, progress=False)
    assert capsys.readouterr().out == ""


def test_center_point(coarse_egg_mesh):
    circle = create_circular_mesh(radius=0.02, mesh_size=0.005)
    solver = HeatSolver(circle, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)
    assert solver.center_point() == (0.0, 0.0)

    solver = HeatSolver(coarse_egg_mesh, EGG_MATERIALS, T_ROOM, T_BOILING, dt=5.0)
    r_c, z_c = solver.center_point()
    assert r_c == 0.0
    assert z_c == pytest.approx(-0.002, abs=1e-12)

    mesh = create_egg_mesh(yolk_center_z=0.003, mesh_size=0.004, mesh_size_boundary=0.002)
    solver = HeatSolver(mesh, EGG_MATERIALS, T_ROOM, T_BOILING, dt=5.0)
    assert solver.center_point()[1] == pytest.approx(0.003, abs=1e-12)


def test_center_temperature_is_interpolated_not_nearest_node():
    """On a mesh with no node at the center, history equals the P1 interpolation there."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=10.0)
    solver.run(100.0, progress=False)
    times, T_center = solver.center_temperature_history()
    assert len(times) == len(T_center) == len(solver.temperatures)
    r, z = solver.center_point()
    expected = [solver.temperature_at(r, z, step=k) - 273.15 for k in range(len(times))]
    assert np.allclose(T_center, expected, atol=1e-9)


def test_temperature_at_interpolates_inside_and_raises_outside():
    R = 0.02
    mesh = create_circular_mesh(radius=R, mesh_size=0.004)
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=10.0)
    solver.run(100.0, progress=False)

    # Centroid of an element: the P1 interpolant is the mean of the vertex values
    nodes = mesh.t[:, 7]
    r_c, z_c = mesh.p[:, nodes].mean(axis=1)
    for step in (-1, 3):
        expected = solver.temperatures[step][nodes].mean()
        assert solver.temperature_at(r_c, z_c, step=step) == pytest.approx(expected, rel=1e-12)

    # At a node, the nodal value
    j = int(solver.interior_dofs[5])
    assert solver.temperature_at(*mesh.p[:, j]) == pytest.approx(solver.T[j], rel=1e-12)

    # A point on the shell, in the middle of the sphere edge, is inside the mesh
    assert solver.temperature_at(0.0, R - 1e-9) > T_ROOM

    for r, z in ((R * 1.01, 0.0), (0.0, R * 1.01), (-0.001, 0.0), (0.05, 0.05)):
        with pytest.raises(ValueError, match="outside the mesh"):
            solver.temperature_at(r, z)


def test_missing_shell_boundary_raises():
    full = create_circular_mesh(radius=0.02, mesh_size=0.005)
    mesh = MeshTri(full.p, full.t).with_boundaries({"axis": lambda x: x[0] < 1e-9})
    assert "shell" not in mesh.boundaries
    with pytest.raises(ValueError, match="shell"):
        HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)


def test_region_min_temperature_history(default_egg_run):
    solver = default_egg_run
    mesh = solver.mesh
    times, white_min = solver.region_min_temperature_history("white")
    _, yolk_min = solver.region_min_temperature_history("yolk")
    assert len(times) == len(white_min) == len(yolk_min) == 801

    # Matches a direct minimum over the nodes of each subdomain
    k = 400
    white_nodes = np.unique(mesh.t[:, mesh.subdomains["white"]])
    yolk_nodes = np.unique(mesh.t[:, mesh.subdomains["yolk"]])
    assert white_min[k] == pytest.approx(solver.temperatures[k][white_nodes].min() - 273.15)
    assert yolk_min[k] == pytest.approx(solver.temperatures[k][yolk_nodes].min() - 273.15)
    # Once heat has penetrated (the first seconds only show round-off-level
    # undershoot), the yolk, deeper inside, is colder than the white
    assert np.all(yolk_min[30:] <= white_min[30:])
    assert np.all(yolk_min[100:] < white_min[100:])
    # Initially both regions are at T_init (the white includes shell nodes at 100 °C)
    assert white_min[0] == pytest.approx(20.0) and yolk_min[0] == pytest.approx(20.0)

    with pytest.raises(ValueError, match="subdomain"):
        solver.region_min_temperature_history("albumen")


def test_region_fraction_above_history_is_volume_weighted():
    """Fractions use the axisymmetric (r-weighted) volume, not area."""
    R = 0.02
    mesh = create_circular_mesh(radius=R, mesh_size=0.0015)
    solver = HeatSolver(mesh, WHITE_CONST, T_ROOM, T_BOILING, dt=1.0)
    # Impose analytic nodal fields in the stored history
    r, z = mesh.p
    solver.temperatures = [np.full(mesh.p.shape[1], 300.0), r.copy(), z.copy()]
    solver.times = [0.0, 1.0, 2.0]

    _, f_const_low = solver.region_fraction_above_history("egg", 301.0)
    _, f_const_all = solver.region_fraction_above_history("egg", 300.0)
    assert f_const_low[0] == 0.0 and f_const_all[0] == 1.0

    # Volume of the sphere with distance from the axis >= a is (1 - (a/R)^2)^(3/2)
    # of the sphere, vs a planar area fraction of about 0.50 for a = R / 2
    _, f_r = solver.region_fraction_above_history("egg", R / 2)
    assert f_r[1] == pytest.approx((1.0 - 0.25) ** 1.5, abs=0.01)  # 0.6495
    # Half of the sphere is above the equatorial plane
    _, f_z = solver.region_fraction_above_history("egg", 0.0)
    assert f_z[2] == pytest.approx(0.5, abs=0.01)

    with pytest.raises(ValueError, match="subdomain"):
        solver.region_fraction_above_history("yolk", 300.0)


def test_solve_condense_dirichlet_system():
    # -u'' = 0 on 4 nodes in a chain with u(0) = 1, u(3) = 4 -> linear profile
    A = csc_matrix(
        np.array(
            [[1.0, -1.0, 0.0, 0.0],
             [-1.0, 2.0, -1.0, 0.0],
             [0.0, -1.0, 2.0, -1.0],
             [0.0, 0.0, -1.0, 1.0]]
        )
    )
    b = np.zeros(4)
    x_bc = np.array([1.0, 0.0, 0.0, 4.0])
    u = solve_condense(A, b, x_bc, np.array([0, 3]))
    assert np.allclose(u, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# 6. Doneness
# ---------------------------------------------------------------------------

def test_crossing_time_helper():
    t = np.array([0.0, 10.0, 20.0, 30.0])
    v = np.array([0.0, 10.0, 30.0, 40.0])
    assert crossing_time(t, v, 5.0) == pytest.approx(5.0)
    assert crossing_time(t, v, 20.0) == pytest.approx(15.0)
    assert crossing_time(t, v, 10.0) == pytest.approx(10.0)  # exactly on a sample
    assert crossing_time(t, v, 0.0) == 0.0  # already reached at the first sample
    assert np.isnan(crossing_time(t, v, 40.5))
    # First upward crossing of a non-monotone series
    assert crossing_time(t, np.array([0.0, 6.0, 2.0, 8.0]), 5.0) == pytest.approx(50.0 / 6.0)
    assert crossing_time(t, np.array([0.0, 4.0, 2.0, 8.0]), 5.0) == pytest.approx(20.0 + 10.0 * 3.0 / 6.0)
    with pytest.raises(ValueError):
        crossing_time(t, v[:3], 1.0)


def test_default_egg_doneness_times(default_egg_run):
    """Pinned doneness times for the default egg (20 °C start, 100 °C bath)."""
    times = doneness_times(default_egg_run)
    assert set(times) == {"soft", "medium", "hard"}
    assert times["soft"] == pytest.approx(343.3, abs=1.5)
    assert times["medium"] == pytest.approx(438.8, abs=1.5)
    assert times["hard"] == pytest.approx(697.6, abs=1.5)
    assert times["soft"] < times["medium"] < times["hard"]


def test_default_egg_center_crossing_times(default_egg_run):
    crossings = center_crossing_times(default_egg_run)
    assert list(crossings) == [63.0, 65.0, 70.0]
    assert crossings[63.0] == pytest.approx(615.4, abs=1.5)
    assert crossings[65.0] == pytest.approx(637.4, abs=1.5)
    assert crossings[70.0] == pytest.approx(697.6, abs=1.5)
    assert crossings[63.0] < crossings[65.0] < crossings[70.0]
    # The yolk center is the coldest yolk point here, so the hard-boiled time
    # (coldest yolk point at 70 °C) equals the center's 70 °C crossing
    assert doneness_times(default_egg_run)["hard"] == pytest.approx(crossings[70.0], abs=0.05)

    custom = center_crossing_times(default_egg_run, thresholds_C=(50.0,))
    assert list(custom) == [50.0]
    assert 300.0 < custom[50.0] < crossings[63.0]


def test_doneness_definitions_are_consistent_with_fields(default_egg_run):
    """At the reported times the field satisfies the definition (and not earlier)."""
    solver = default_egg_run
    times = doneness_times(solver)
    t_w, white_min = solver.region_min_temperature_history("white")
    _, yolk_min = solver.region_min_temperature_history("yolk")
    _, frac = solver.region_fraction_above_history("yolk", YOLK_FULL_SET_K)

    k = int(np.ceil(times["soft"]))
    assert white_min[k] >= WHITE_DENATURATION_K - 273.15 > white_min[k - 2]
    k = int(np.ceil(times["medium"]))
    assert frac[k] >= 0.5 > frac[k - 2]
    k = int(np.ceil(times["hard"]))
    assert yolk_min[k] >= YOLK_FULL_SET_K - 273.15 > yolk_min[k - 2]


def test_doneness_not_reached_is_nan(coarse_egg_mesh):
    solver = HeatSolver(coarse_egg_mesh, EGG_MATERIALS, T_ROOM, T_BOILING, dt=5.0)
    solver.run(100.0, progress=False)
    times = doneness_times(solver)
    assert all(np.isnan(v) for v in times.values())
    assert np.isnan(center_crossing_times(solver)[63.0])


def test_doneness_requires_white_and_yolk_subdomains():
    circle = create_circular_mesh(radius=0.02, mesh_size=0.005)
    solver = HeatSolver(circle, WHITE_CONST, T_ROOM, T_BOILING, dt=5.0)
    solver.run(10.0, progress=False)
    with pytest.raises(ValueError, match="white.*yolk"):
        doneness_times(solver)


def test_doneness_fridge_slower_than_room_and_latent_heat_slows(coarse_egg_mesh):
    """Colder start and endothermic denaturation both delay every doneness level."""
    def run(T_init, materials):
        solver = HeatSolver(coarse_egg_mesh, materials, T_init, T_BOILING, dt=5.0)
        solver.run(1000.0, progress=False)
        return doneness_times(solver)

    room = run(T_ROOM, EGG_MATERIALS)
    fridge = run(T_FRIDGE, EGG_MATERIALS)
    room_latent = run(T_ROOM, EGG_MATERIALS_DENAT)

    for level in ("soft", "medium", "hard"):
        assert np.isfinite(room[level]) and np.isfinite(fridge[level]) and np.isfinite(room_latent[level])
        assert fridge[level] > room[level] + 30.0
        assert room_latent[level] > room[level] + 2.0

    # Measured on this coarse mesh (dt = 5 s):
    #   room:   348.3 / 438.4 / 702.6 s
    #   fridge: 394.1 / 493.1 / 760.6 s
    #   latent: 352.0 / 442.5 / 712.1 s
    assert room["hard"] == pytest.approx(702.6, abs=5.0)
    assert fridge["hard"] == pytest.approx(760.6, abs=5.0)
    assert room_latent["hard"] == pytest.approx(712.1, abs=5.0)
