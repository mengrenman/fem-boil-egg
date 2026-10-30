"""Unit and integration tests for egg_fem."""

import glob
import os
import tempfile
import numpy as np
import pytest
from scipy.integrate import quad

from egg_fem import (
    EggMaterial,
    HeatSolver,
    H_BOILING_SHELL,
    T_BOILING,
    T_FRIDGE,
    T_ROOM,
    WHITE_CONST,
    WHITE_DENATURATION_K,
    YOLK_CONST,
    YOLK_DENATURATION_K,
    YOLK_FULL_SET_K,
    create_circular_mesh,
    create_egg_mesh,
    egg_profile,
    egg_white_denaturation_properties,
    egg_white_properties,
    egg_yolk_denaturation_properties,
    egg_yolk_properties,
)


# ---------------------------------------------------------------------------
# 1. Material Property Tests
# ---------------------------------------------------------------------------

def test_material_properties_bounds():
    """Verify temperature-dependent material properties across [0 °C, 100 °C]."""
    T = np.linspace(273.15, 373.15, 50)

    for mat in (egg_white_properties, egg_yolk_properties, WHITE_CONST, YOLK_CONST):
        rho = mat.density(T)
        kappa = mat.conductivity(T)
        cp = mat.specific_heat(T)
        rho_cp = mat.rho_cp(T)

        assert np.all(rho > 900.0) and np.all(rho < 1200.0)
        assert np.all(kappa > 0.2) and np.all(kappa < 1.0)
        assert np.all(cp > 2500.0) and np.all(cp < 5000.0)
        assert np.allclose(rho_cp, rho * cp)


def test_latent_heat_denaturation():
    """Verify effective specific heat integrates to base heat + latent heat."""
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
# 2. Geometry & Mesh Generation Tests
# ---------------------------------------------------------------------------

def test_egg_profile():
    """Verify egg profile coordinates and symmetry endpoints."""
    r, z = egg_profile(n=100, length=0.055, breadth=0.042)
    assert r.shape == (100,)
    assert z.shape == (100,)
    assert np.all(r >= 0.0)
    assert np.isclose(r[0], 0.0, atol=1e-12)
    assert np.isclose(r[-1], 0.0, atol=1e-12)
    assert np.isclose(z[0] - z[-1], 0.055, rtol=1e-3)
    assert np.isclose(np.max(r), 0.021, rtol=1e-2)


def test_circular_mesh_and_tempfile_cleanup():
    """Verify circular mesh creation and ensure no temporary .msh files leak."""
    tmp_dir = tempfile.gettempdir()
    before_files = set(glob.glob(os.path.join(tmp_dir, "*.msh")))

    mesh = create_circular_mesh(radius=0.02, mesh_size=0.005)

    after_files = set(glob.glob(os.path.join(tmp_dir, "*.msh")))
    assert after_files == before_files, "Temporary .msh file was not cleaned up!"

    assert "shell" in mesh.boundaries
    assert "axis" in mesh.boundaries
    assert mesh.p.shape[1] > 10


def test_egg_mesh_subdomains():
    """Verify realistic egg mesh contains yolk and white subdomains and boundaries."""
    mesh = create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002)
    assert "shell" in mesh.boundaries
    assert "axis" in mesh.boundaries
    assert "white" in mesh.subdomains
    assert "yolk" in mesh.subdomains
    assert len(mesh.subdomains["white"]) > 0
    assert len(mesh.subdomains["yolk"]) > 0


# ---------------------------------------------------------------------------
# 3. Solver Physics & Boundary Condition Tests
# ---------------------------------------------------------------------------

def test_dirichlet_solver_monotonic_and_bounded():
    """Verify Dirichlet solver heats center towards T_BOILING."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.003)
    solver = HeatSolver(mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0)
    times, temps = solver.run(t_end=600.0, progress=False)

    assert len(times) == 121
    for T in temps:
        # Consistent P1 mass matrix can have slight (<2 K) initial undershoot at step 1
        assert np.all(T >= T_ROOM - 3.0)
        assert np.all(T <= T_BOILING + 1e-6)
        assert np.allclose(T[solver.shell_dofs], T_BOILING)

    _, T_center = solver.center_temperature_history()
    # Once thermal wave reaches the center (>150 s), center temperature increases monotonically
    assert np.all(np.diff(T_center[30:]) > 0.0)
    assert T_center[-1] > 60.0


def test_robin_boundary_condition():
    """Verify Robin BC produces realistic finite boundary heat transfer."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.004)

    solver_dirichlet = HeatSolver(
        mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0, h_conv=None
    )
    solver_robin = HeatSolver(
        mesh,
        WHITE_CONST,
        T_init=T_ROOM,
        T_boundary=T_BOILING,
        dt=5.0,
        h_conv=H_BOILING_SHELL,
    )
    solver_insulated = HeatSolver(
        mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0, h_conv=0.0
    )

    solver_dirichlet.run(t_end=300.0, progress=False)
    solver_robin.run(t_end=300.0, progress=False)
    solver_insulated.run(t_end=300.0, progress=False)

    # Insulated boundary (h_conv = 0) keeps temperature constant at T_ROOM
    assert np.allclose(solver_insulated.T, T_ROOM, atol=1e-6)

    # Robin boundary heats up the shell towards T_BOILING, but remains strictly < T_BOILING
    T_shell_robin = solver_robin.T[solver_robin.shell_dofs]
    assert np.all(T_shell_robin > T_ROOM)
    assert np.all(T_shell_robin < T_BOILING)

    # Center temperature with finite h_conv lags slightly behind infinite h_conv (Dirichlet)
    _, Tc_dirichlet = solver_dirichlet.center_temperature_history()
    _, Tc_robin = solver_robin.center_temperature_history()
    assert Tc_robin[-1] > (T_ROOM - 273.15)
    assert Tc_robin[-1] < Tc_dirichlet[-1]


def test_robin_converges_to_dirichlet_for_large_h():
    """As h_conv -> infinity, Robin BC solution converges to Dirichlet BC solution."""
    mesh = create_circular_mesh(radius=0.02, mesh_size=0.004)

    solver_dir = HeatSolver(
        mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=2.0, h_conv=None
    )
    solver_high_h = HeatSolver(
        mesh, WHITE_CONST, T_init=T_ROOM, T_boundary=T_BOILING, dt=2.0, h_conv=1e8
    )
    # Match initial shell state so both start from the exact same T^0 vector
    solver_high_h.T[solver_high_h.shell_dofs] = T_BOILING
    solver_high_h.temperatures[0] = solver_high_h.T.copy()

    solver_dir.run(t_end=200.0, progress=False)
    solver_high_h.run(t_end=200.0, progress=False)

    _, Tc_dir = solver_dir.center_temperature_history()
    _, Tc_high_h = solver_high_h.center_temperature_history()
    assert np.allclose(Tc_dir, Tc_high_h, atol=1e-3)
    assert np.allclose(solver_dir.T, solver_high_h.T, atol=1e-3)


def test_latent_heat_slows_heating():
    """Endothermic latent heat of denaturation slows center heating slightly."""
    mesh = create_egg_mesh(mesh_size=0.004, mesh_size_boundary=0.002)

    mat_std = {"white": egg_white_properties, "yolk": egg_yolk_properties}
    mat_denat = {
        "white": egg_white_denaturation_properties,
        "yolk": egg_yolk_denaturation_properties,
    }

    solver_std = HeatSolver(mesh, mat_std, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0)
    solver_denat = HeatSolver(
        mesh, mat_denat, T_init=T_ROOM, T_boundary=T_BOILING, dt=5.0
    )

    solver_std.run(t_end=600.0, progress=False)
    solver_denat.run(t_end=600.0, progress=False)

    _, Tc_std = solver_std.center_temperature_history()
    _, Tc_denat = solver_denat.center_temperature_history()

    # At 600 s (~63 °C center), endothermic denaturation absorbs heat so Tc_denat < Tc_std
    assert Tc_denat[-1] < Tc_std[-1]
    assert Tc_std[-1] - Tc_denat[-1] < 5.0  # physically reasonable difference

