"""FEM solver for the axisymmetric transient heat equation.

Weak form in cylindrical coordinates (r, z) with axial symmetry:

    ∫ ρCp ∂T/∂t · v · r  dr dz  +  ∫ κ ∇T·∇v · r  dr dz  =  0

Time integration uses backward Euler (implicit, unconditionally stable).
Material properties are evaluated at the previous time-step temperature
(lagged coefficients), giving a linear system at each step.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, FacetBasis, ElementTriP1, BilinearForm, LinearForm, asm
from skfem.helpers import dot, grad


# ---------------------------------------------------------------------------
# Bilinear and linear forms
# ---------------------------------------------------------------------------

@BilinearForm
def mass_axisym(u, v, w):
    """Mass matrix:  ∫ ρCp · u · v · r  dr dz."""
    r = w.x[0]
    return w["rho_cp"] * u * v * np.maximum(r, 1e-12)


@BilinearForm
def stiffness_axisym(u, v, w):
    """Stiffness matrix:  ∫ κ · ∇u·∇v · r  dr dz."""
    r = w.x[0]
    return w["kappa"] * dot(grad(u), grad(v)) * np.maximum(r, 1e-12)


@BilinearForm
def robin_lhs_axisym(u, v, w):
    """Boundary Robin matrix:  ∫_Γ h · u · v · r  ds."""
    r = w.x[0]
    return w["h"] * u * v * np.maximum(r, 1e-12)


@LinearForm
def robin_rhs_axisym(v, w):
    """Boundary Robin vector:  ∫_Γ h · T_boundary · v · r  ds."""
    r = w.x[0]
    return w["h"] * w["T_boundary"] * v * np.maximum(r, 1e-12)


# Sentinel: "argument omitted" for set_boundary_conditions (None already means
# "Dirichlet").
_KEEP = object()

# Simulated-time interval between progress lines printed by HeatSolver.run [s].
_PROGRESS_INTERVAL = 60.0

# Quadrature order of the basis used to evaluate volume fractions above a
# threshold (16 points per triangle).  The 3-point rule used for assembly
# resolves an isotherm cutting through an element too coarsely.  The indicator
# integrand is discontinuous, so the quadrature error does not decrease
# monotonically: for the default egg the medium (50% of the yolk at 70 °C) time
# varies by about 2 s over intorder 2 to 12 (438.1, 440.3, 439.3, 438.8, 438.7,
# 439.3 s at intorder 2, 4, 6, 8, 10, 12).  Order 8 is a compromise; treat the
# crossing times as good to about 1 s from this source.
_FRACTION_INTORDER = 8


def _next_report_time(time):
    """First multiple of the progress interval strictly after *time* [s]."""
    return (np.floor(time / _PROGRESS_INTERVAL + 1e-6) + 1.0) * _PROGRESS_INTERVAL


def _validate_h_conv(h_conv):
    """Return ``h_conv`` as float (or None); raise ValueError if negative or not finite."""
    if h_conv is None:
        return None
    h = float(h_conv)
    if not np.isfinite(h) or h < 0.0:
        raise ValueError(f"h_conv must be None or a finite number >= 0, got {h_conv!r}")
    return h


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class HeatSolver:
    """Solves transient heat conduction in an axisymmetric egg domain.

    Parameters
    ----------
    mesh : skfem.MeshTri
        Finite element mesh (in the r-z half-plane) with a "shell" boundary.
    material : EggMaterial or dict
        Single material, or ``{"yolk": mat_yolk, "white": mat_white}``
        for multi-region problems.  When a dict is passed every key must be
        a subdomain of the mesh and every element must belong to exactly one
        of the listed subdomains; otherwise ``ValueError`` is raised.
    T_init : float
        Initial temperature [K].
    T_boundary : float
        Ambient or Dirichlet temperature on the "shell" boundary [K].
    dt : float
        Time step [s]; must be positive.
    h_conv : float or None, optional
        Convective / effective shell heat transfer coefficient [W/(m²·K)].
        If ``None`` (default), a Dirichlet condition ``T = T_boundary`` is
        applied on the shell boundary.  If a float ``>= 0`` is given, a
        Robin boundary condition ``-κ ∂T/∂n = h_conv (T - T_boundary)`` is
        applied on the shell boundary (``0`` is an insulated shell).

    Notes
    -----
    Initial-condition convention.  With a Dirichlet shell the t = 0 state
    already has the shell nodes at ``T_boundary`` and all other nodes at
    ``T_init``: the compatible initial state, whose shell values agree with
    the boundary data (standard practice).  With a Robin shell the field
    starts uniform at ``T_init``, because the shell temperature is continuous
    in time and rises from ``T_init`` as heat flows through the surface film.
    The two conventions differ only by a discretization-level offset: a
    Robin run with a huge ``h_conv`` and a Dirichlet run on the same mesh
    differ at 300 s in the center temperature by 1.7, 0.47, 0.13 and 0.03 K
    for 4, 2, 1 and 0.5 mm elements (sphere, ``dt = 1 s``), so they converge
    to the same solution as the mesh is refined.

    Consistent (non-lumped) P1 mass matrix.  In the first steps after the
    start, the discrete solution undershoots ``T_init`` slightly in the nodes
    next to the shell, because the sharp initial gradient is not resolved
    (backward Euler with a consistent mass matrix does not satisfy the
    discrete maximum principle there).  Measured values for a 20 °C start in
    a 100 °C Dirichlet bath: about 1.4 K for the 1.5 mm sphere mesh with
    ``dt = 1 s``, below 0.05 K for the default egg mesh (0.6 mm elements at
    the shell) with ``dt = 1 s``, and up to 2.8 K on a 3 mm mesh.  The
    undershoot is largest next to the shell, is gone within about 10-40 s,
    and shrinks as the mesh and the time step are refined.  Its effect on the
    heating curve is small: for the sphere benchmark in ``tests/test_fem.py``
    the 63 °C crossing time of the center is within 0.4 s of the exact value.

    State.  ``times`` and ``temperatures`` hold the history (starting with
    t = 0); ``T`` is the current field and is always the last entry of
    ``temperatures``.  ``set_boundary_conditions`` changes the boundary data
    for subsequent steps only and never modifies the stored fields.
    """

    def __init__(self, mesh, material, T_init, T_boundary, dt, h_conv=None):
        dt = float(dt)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be a positive finite number, got {dt!r}")
        h_conv = _validate_h_conv(h_conv)

        if "shell" not in (mesh.boundaries or {}):
            raise ValueError('the mesh must have a boundary named "shell"')

        self.mesh = mesh
        self.elem = ElementTriP1()
        self.basis = Basis(mesh, self.elem)
        self.dt = dt
        self.T_boundary = float(T_boundary)
        self.h_conv = h_conv

        # DOFs on the shell boundary & interior DOFs
        self.shell_dofs = self.basis.get_dofs("shell").all()
        self.interior_dofs = np.setdiff1d(
            np.arange(self.basis.N, dtype=int), self.shell_dofs
        )
        self.shell_fbasis = FacetBasis(
            mesh, self.elem, facets=mesh.boundaries["shell"]
        )

        # Material setup
        self._multi_region = isinstance(material, dict)
        if self._multi_region:
            self.materials = material
            self._elem_material = self._build_element_material_map()
        else:
            self.material = material

        # State.  The shell nodes start at T_boundary for a Dirichlet shell
        # (compatible initial state) and at T_init for a Robin shell; see the
        # class docstring.
        self.T = np.full(self.basis.N, float(T_init))
        if self.h_conv is None:
            self.T[self.shell_dofs] = self.T_boundary
        self.set_boundary_conditions(self.T_boundary, self.h_conv)
        self.time = 0.0

        # Storage
        self.times = [0.0]
        self.temperatures = [self.T.copy()]

        self._center_probe = None
        self._fraction_basis = None

    def set_boundary_conditions(self, T_boundary, h_conv=_KEEP):
        """Update the shell boundary temperature and heat transfer coefficient.

        Useful for multi-stage simulations (e.g. transferring a boiled egg to
        an ice-water bath or resting in ambient air).  The new conditions
        apply from the next time step on; the stored fields (``T`` and
        ``temperatures``) are not modified.

        Parameters
        ----------
        T_boundary : float
            New shell / ambient temperature [K].
        h_conv : float or None, optional
            If omitted, the current boundary-condition type is kept (only
            the temperature changes).  ``None`` switches to a Dirichlet
            condition ``T = T_boundary``; a float ``>= 0`` switches to (or
            updates) a Robin condition with that coefficient [W/(m²·K)].
        """
        if h_conv is _KEEP:
            h_conv = self.h_conv
        h_conv = _validate_h_conv(h_conv)
        T_boundary = float(T_boundary)

        self.T_boundary = T_boundary
        self.h_conv = h_conv

        if self.h_conv is not None:
            self._K_robin = asm(
                robin_lhs_axisym, self.shell_fbasis, h=self.h_conv
            )
            self._f_robin = asm(
                robin_rhs_axisym,
                self.shell_fbasis,
                h=self.h_conv,
                T_boundary=self.T_boundary,
            )
        else:
            self._K_robin = None
            self._f_robin = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_element_material_map(self):
        """Return an array mapping each element to a key in self.materials.

        Raises ValueError unless every key is a subdomain of the mesh and
        every element belongs to exactly one of the listed subdomains.
        """
        subdomains = self.mesh.subdomains
        if not subdomains:
            raise ValueError(
                "a dict of materials requires a mesh with named subdomains, "
                "but the mesh has none"
            )
        missing = sorted(set(self.materials) - set(subdomains))
        if missing:
            raise ValueError(
                f"material keys {missing} are not subdomains of the mesh "
                f"(mesh subdomains: {sorted(subdomains)})"
            )
        n_elem = self.mesh.t.shape[1]
        count = np.zeros(n_elem, dtype=int)
        mat_map = np.full(n_elem, "", dtype=object)
        for name in self.materials:
            idx = np.unique(np.asarray(subdomains[name], dtype=int))
            count[idx] += 1
            mat_map[idx] = name
        n_uncovered = int(np.sum(count == 0))
        if n_uncovered:
            raise ValueError(
                f"{n_uncovered} of {n_elem} mesh elements are not in any of the "
                f"subdomains {sorted(self.materials)}; give a material for every "
                f"region of the mesh"
            )
        n_multiple = int(np.sum(count > 1))
        if n_multiple:
            raise ValueError(
                f"{n_multiple} mesh elements belong to more than one of the "
                f"subdomains {sorted(self.materials)}"
            )
        return mat_map

    def _eval_properties(self):
        """Evaluate ρCp and κ at quadrature points using current temperature.

        Returns arrays matching the shape of the interpolated field
        (n_elements, n_qp) in scikit-fem's convention.
        """
        T_qp = np.asarray(self.basis.interpolate(self.T))  # values at quad pts

        if not self._multi_region:
            rho_cp = self.material.rho_cp(T_qp)
            kappa = self.material.conductivity(T_qp)
        else:
            rho_cp = np.empty_like(T_qp)
            kappa = np.empty_like(T_qp)
            for name, mat in self.materials.items():
                mask = self._elem_material == name
                if np.any(mask):
                    rho_cp[mask] = mat.rho_cp(T_qp[mask])
                    kappa[mask] = mat.conductivity(T_qp[mask])
        return rho_cp, kappa

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self):
        """Advance one time step (backward Euler)."""
        rho_cp, kappa = self._eval_properties()

        M = asm(mass_axisym, self.basis, rho_cp=rho_cp)
        K = asm(stiffness_axisym, self.basis, kappa=kappa)

        if self.h_conv is not None:
            A = M / self.dt + K + self._K_robin
            b = M @ self.T / self.dt + self._f_robin
            self.T = spsolve(A.tocsc(), b)
        else:
            A = M / self.dt + K
            b = M @ self.T / self.dt

            # Dirichlet BC: T = T_boundary on shell
            bc_vec = np.zeros_like(self.T)
            bc_vec[self.shell_dofs] = self.T_boundary

            self.T = solve_condense(
                A, b, bc_vec, self.shell_dofs, interior=self.interior_dofs
            )

        self.time += self.dt

        self.times.append(self.time)
        self.temperatures.append(self.T.copy())

    def run(self, t_end, progress=True):
        """Run the simulation until *t_end* seconds.

        Whole steps of ``dt`` are taken until ``time >= t_end``, so the final
        time can exceed *t_end* by less than one step when ``t_end`` is not a
        multiple of ``dt``.  The step count is robust to floating-point
        round-off in the accumulated time, so ``run(10)`` followed by
        ``run(20)`` with ``dt = 1`` takes exactly 20 steps in total.

        Parameters
        ----------
        t_end : float
            End time [s].
        progress : bool
            If True, print a progress line every 60 s of simulated time
            (independent of ``dt``).

        Returns
        -------
        times : list[float]
        temperatures : list[ndarray]
        """
        # The tolerance absorbs round-off in self.time (accumulated one dt at
        # a time); it is far smaller than any meaningful fraction of a step.
        n_steps = max(0, int(np.ceil((t_end - self.time) / self.dt - 1e-6)))
        next_report = _next_report_time(self.time)

        for _ in range(n_steps):
            self.step()
            if progress and self.time >= next_report - 1e-6 * self.dt:
                T_center = self._center_probe_matrix() @ self.T
                print(
                    f"  t = {self.time:6.1f} s "
                    f"({self.time / 60:.1f} min)  "
                    f"T_center = {T_center[0] - 273.15:.1f} °C"
                )
                next_report = _next_report_time(self.time)

        return self.times, self.temperatures

    # ------------------------------------------------------------------
    # Probes and diagnostics
    # ------------------------------------------------------------------

    def center_point(self):
        """Return ``(r, z)`` of the egg's center, ``(0.0, z_c)``.

        ``z_c`` is the yolk center if the mesh has a "yolk" subdomain (the
        midpoint of the yolk's nodes on the symmetry axis), else 0.
        """
        z_c = 0.0
        subdomains = self.mesh.subdomains
        if subdomains and "yolk" in subdomains:
            coords = self.mesh.p
            yolk_nodes = np.unique(self.mesh.t[:, subdomains["yolk"]])
            on_axis = yolk_nodes[coords[0, yolk_nodes] < 1e-6 * np.max(coords[0])]
            if len(on_axis) == 0:
                raise ValueError("the yolk subdomain has no nodes on the symmetry axis")
            z_c = 0.5 * (np.min(coords[1, on_axis]) + np.max(coords[1, on_axis]))
        return (0.0, float(z_c))

    def _center_probe_matrix(self):
        """Sparse (1, N) matrix interpolating a nodal field at the center point."""
        if self._center_probe is None:
            r, z = self.center_point()
            self._center_probe = self.basis.probes(np.array([[r], [z]]))
        return self._center_probe

    def temperature_at(self, r, z, step=-1):
        """Interpolate the temperature at a single point (r, z).

        Parameters
        ----------
        r, z : float
            Coordinates [m].
        step : int
            Time-step index (default: latest).

        Returns
        -------
        float
            Temperature [K].

        Raises
        ------
        ValueError
            If the point lies outside the mesh.
        """
        T = self.temperatures[step]
        try:
            probe = self.basis.probes(np.array([[float(r)], [float(z)]]))
        except ValueError as exc:
            raise ValueError(
                f"point (r={r}, z={z}) is outside the mesh; cannot interpolate "
                f"the temperature there"
            ) from exc
        return float((probe @ T)[0])

    def center_temperature_history(self):
        """Return ``(times, T_center)`` arrays, with T_center in °C.

        The temperature is interpolated (P1) at :meth:`center_point`.
        """
        probe = self._center_probe_matrix()
        T_all = np.asarray(self.temperatures)  # (n_times, N)
        T_center = np.asarray(probe @ T_all.T).ravel() - 273.15
        return np.array(self.times), T_center

    def _region_elements(self, region):
        """Element indices of the named subdomain (ValueError if unknown)."""
        subdomains = self.mesh.subdomains
        if not subdomains or region not in subdomains:
            available = sorted(subdomains) if subdomains else []
            raise ValueError(
                f"mesh has no subdomain {region!r} (available: {available})"
            )
        return np.asarray(subdomains[region], dtype=int)

    def region_min_temperature_history(self, region):
        """Coldest temperature in a subdomain over time.

        Because the field is P1, its minimum over a subdomain is attained at
        one of the subdomain's nodes, so this is the exact minimum.

        Parameters
        ----------
        region : str
            Subdomain name, e.g. "white" or "yolk".

        Returns
        -------
        times : ndarray
            Time [s].
        T_min_C : ndarray
            Minimum temperature over the subdomain's nodes [°C].
        """
        elements = self._region_elements(region)
        nodes = np.unique(self.mesh.t[:, elements])
        T_all = np.asarray(self.temperatures)
        return np.array(self.times), T_all[:, nodes].min(axis=1) - 273.15

    def region_fraction_above_history(self, region, T_threshold_K):
        """Volume fraction of a subdomain at or above a temperature, over time.

        The fraction is axisymmetric (``r``-weighted, i.e. true volume, not
        area).  The P1 field is interpolated to the points of a 16-point
        quadrature rule on every element of the subdomain, and the fraction
        is the weighted share of points at or above the threshold; the
        rule is finer than the one used for assembly so that an isotherm
        cutting through elements is resolved well.

        Parameters
        ----------
        region : str
            Subdomain name, e.g. "yolk".
        T_threshold_K : float
            Threshold temperature [K].

        Returns
        -------
        times : ndarray
            Time [s].
        fraction : ndarray
            Volume fraction in [0, 1] with ``T >= T_threshold_K``.
        """
        elements = self._region_elements(region)
        if self._fraction_basis is None:
            self._fraction_basis = Basis(
                self.mesh, self.elem, intorder=_FRACTION_INTORDER
            )
        basis = self._fraction_basis
        r = np.asarray(basis.global_coordinates()[0])[elements]
        weights = np.asarray(basis.dx)[elements] * r  # (n_el, n_qp), r-weighted
        total = weights.sum()
        fraction = np.empty(len(self.temperatures))
        for k, T in enumerate(self.temperatures):
            T_qp = np.asarray(basis.interpolate(T))[elements]
            fraction[k] = weights[T_qp >= T_threshold_K].sum() / total
        return np.array(self.times), fraction


# ---------------------------------------------------------------------------
# Linear algebra helper
# ---------------------------------------------------------------------------

def solve_condense(A, b, x_bc, dofs, interior=None):
    """Solve A x = b with Dirichlet conditions x[dofs] = x_bc[dofs]."""
    u = x_bc.copy()
    if interior is None:
        interior = np.setdiff1d(np.arange(A.shape[0]), dofs)
    b_int = b[interior] - A[interior][:, dofs] @ x_bc[dofs]
    A_int = A[interior][:, interior]
    u[interior] = spsolve(A_int.tocsc(), b_int)
    return u
