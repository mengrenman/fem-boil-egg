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

from .materials import EggMaterial


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


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class HeatSolver:
    """Solves transient heat conduction in an axisymmetric egg domain.

    Parameters
    ----------
    mesh : skfem.MeshTri
        Finite element mesh (in the r-z half-plane).
    material : EggMaterial or dict
        Single material, or ``{"yolk": mat_yolk, "white": mat_white}``
        for multi-region problems.  When a dict is passed the mesh must
        contain subdomains named "yolk" and "white".
    T_init : float
        Initial temperature [K].
    T_boundary : float
        Ambient or Dirichlet temperature on the "shell" boundary [K].
    dt : float
        Time step [s].
    h_conv : float or None, optional
        Convective / effective shell heat transfer coefficient [W/(m²·K)].
        If ``None`` (default), a Dirichlet condition ``T = T_boundary`` is
        applied on the shell boundary.  If a positive float is given, a
        Robin boundary condition ``-κ ∂T/∂n = h_conv (T - T_boundary)`` is
        applied on the shell boundary.
    """

    def __init__(self, mesh, material, T_init, T_boundary, dt, h_conv=None):
        self.mesh = mesh
        self.elem = ElementTriP1()
        self.basis = Basis(mesh, self.elem)
        self.dt = dt
        self.T_boundary = T_boundary
        self.h_conv = h_conv

        # DOFs on the shell boundary
        self.shell_dofs = self.basis.get_dofs("shell").all()

        # Pre-assemble Robin boundary contributions if h_conv is provided
        if self.h_conv is not None:
            self.shell_fbasis = FacetBasis(
                mesh, self.elem, facets=mesh.boundaries["shell"]
            )
            self._K_robin = asm(
                robin_lhs_axisym, self.shell_fbasis, h=float(self.h_conv)
            )
            self._f_robin = asm(
                robin_rhs_axisym,
                self.shell_fbasis,
                h=float(self.h_conv),
                T_boundary=float(self.T_boundary),
            )
        else:
            self.shell_fbasis = None
            self._K_robin = None
            self._f_robin = None

        # Material setup
        self._multi_region = isinstance(material, dict)
        if self._multi_region:
            self.materials = material
            # Build element -> material map
            self._elem_material = self._build_element_material_map()
        else:
            self.material = material

        # State
        self.T = np.full(self.basis.N, float(T_init))
        if self.h_conv is None:
            self.T[self.shell_dofs] = T_boundary
        self.time = 0.0

        # Storage
        self.times = [0.0]
        self.temperatures = [self.T.copy()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_element_material_map(self):
        """Return an array mapping each element to 'yolk' or 'white'."""
        n_elem = self.mesh.t.shape[1]
        mat_map = np.empty(n_elem, dtype=object)
        mat_map[:] = "white"  # default
        if "yolk" in self.mesh.subdomains:
            mat_map[self.mesh.subdomains["yolk"]] = "yolk"
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
                if name in self.mesh.subdomains:
                    elems = self.mesh.subdomains[name]
                    rho_cp[elems] = mat.rho_cp(T_qp[elems])
                    kappa[elems] = mat.conductivity(T_qp[elems])
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

            self.T = solve_condense(A, b, bc_vec, self.shell_dofs)

        self.time += self.dt

        self.times.append(self.time)
        self.temperatures.append(self.T.copy())

    def run(self, t_end, progress=True):
        """Run the simulation until *t_end* seconds.

        Parameters
        ----------
        t_end : float
            End time [s].
        progress : bool
            If True, print a progress line every 60 s of simulated time.

        Returns
        -------
        times : list[float]
        temperatures : list[ndarray]
        """
        n_steps = int(np.ceil((t_end - self.time) / self.dt))
        report_interval = max(1, int(60.0 / self.dt))  # every 60 s sim-time

        for i in range(n_steps):
            self.step()
            if progress and (i + 1) % report_interval == 0:
                T_center = self.T[self._center_dof()]
                print(
                    f"  t = {self.time:6.1f} s "
                    f"({self.time / 60:.1f} min)  "
                    f"T_center = {T_center - 273.15:.1f} °C"
                )

        return self.times, self.temperatures

    def _center_dof(self):
        """DOF index closest to (r, z) = (0, 0)."""
        coords = self.basis.mesh.p  # (2, n_nodes)
        dist = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
        return int(np.argmin(dist))

    def temperature_at(self, r, z, step=-1):
        """Interpolate temperature at a single point.

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
        """
        T = self.temperatures[step]
        # Simple inverse-distance weighted interpolation
        coords = self.mesh.p  # (2, n_nodes)
        dist = np.sqrt((coords[0] - r) ** 2 + (coords[1] - z) ** 2)
        # Use 4 nearest nodes
        idx = np.argsort(dist)[:4]
        d = dist[idx]
        d = np.maximum(d, 1e-15)
        w = 1.0 / d
        return float(np.dot(w, T[idx]) / w.sum())

    def center_temperature_history(self):
        """Return (times, T_center) arrays in °C."""
        idx = self._center_dof()
        T_center = np.array([T[idx] for T in self.temperatures]) - 273.15
        return np.array(self.times), T_center


# ---------------------------------------------------------------------------
# Linear algebra helper
# ---------------------------------------------------------------------------

def solve_condense(A, b, x_bc, dofs):
    """Solve A x = b with Dirichlet conditions x[dofs] = x_bc[dofs]."""
    u = x_bc.copy()
    interior = np.setdiff1d(np.arange(A.shape[0]), dofs)
    b_int = b[interior] - A[interior][:, dofs] @ x_bc[dofs]
    A_int = A[interior][:, interior]
    u[interior] = spsolve(A_int.tocsc(), b_int)
    return u
