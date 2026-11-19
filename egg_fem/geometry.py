"""Mesh generation for axisymmetric egg models.

Creates 2D meshes in the (r, z) half-plane for use with an axisymmetric
formulation.  The axis of symmetry is r = 0 (the left edge).

Two mesh generators are provided:
  - create_circular_mesh:  semicircular cross-section (single region); the
                           solid of revolution is a sphere
  - create_egg_mesh:       egg-shaped cross-section with separate yolk and
                           white regions

Both return a scikit-fem MeshTri with named boundaries and subdomains.
``mesh_volume`` integrates the volume of the solid of revolution (the whole
mesh or one named subdomain).

gmsh usage
----------
Both builders open their own gmsh session and clean up after themselves.
If the caller already initialized gmsh, the existing session is left alone:
the builders add a uniquely named model, and afterwards remove only that
model and restore the caller's current model and ``General.Verbosity``.

gmsh installs signal handlers when it is initialized, which Python only
allows on the main thread.  With gmsh >= 4.12 the builders pass
``interruptible=False`` and can run off the main thread; with older gmsh
versions they must be called from the main thread.
"""

import contextlib
import inspect
import io
import os
import tempfile
import uuid

import numpy as np
import gmsh
import meshio
from skfem.io import from_meshio


# ---------------------------------------------------------------------------
# Egg profile parameterization
# ---------------------------------------------------------------------------

# Beyond this the Hugelschaffer formula is no longer egg-like (the egg becomes
# a cone-like shape); the formula itself is valid for asymmetry < 0.5.
_MAX_ASYMMETRY = 0.25


def egg_profile(n=200, length=0.055, breadth=0.042, asymmetry=0.06):
    """Return (r, z) points tracing the right-half egg profile.

    The profile is the Hugelschaffer egg curve

        r(z) = (B / 2) * sqrt((L**2 - 4 z**2) / (L**2 + 8 w z + 4 w**2)),
        z in [-L/2, L/2],   w = asymmetry * L,

    with L the pole-to-pole length and B the maximum breadth.  The pointed
    end is at +z (top) and the blunt end at -z (bottom).  The maximum radius
    is exactly B / 2 and occurs at z = -w, i.e. on the blunt side of the
    midpoint; for ``asymmetry = 0`` the curve is an ellipse.  The pole-to-pole
    length is exactly L because z runs over [-L/2, L/2].

    The points are parameterized as z = (L / 2) cos(t) with t uniformly
    spaced on [0, pi], so z is strictly decreasing along the profile (no
    pole dimples) and the points cluster near the poles, where r(z) has a
    vertical tangent.  The first point is the top pole and the last point
    is the bottom pole, with r = 0 there exactly.

    For the default 55 x 42 mm egg the volume is about 50.7 mL, the maximum
    breadth is at z = -3.3 mm, and the radius is 13.2 mm at z = +20 mm
    versus 15.7 mm at z = -20 mm.

    Parameters
    ----------
    n : int
        Number of points on the profile (at least 3).
    length : float
        Egg length pole-to-pole [m].
    breadth : float
        Maximum egg diameter [m].
    asymmetry : float
        Offset of the maximum-breadth plane from the midpoint as a fraction
        of the length, ``w / L``.  Must satisfy ``0 <= asymmetry < 0.25``
        (the formula degenerates to a cone-like shape beyond that).  Typical
        hen eggs are about 0.05-0.07.

    Returns
    -------
    r, z : ndarray, shape (n,)
        Radial and axial coordinates [m], from the top pole (z = +L/2) to
        the bottom pole (z = -L/2).
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    if not (length > 0.0 and breadth > 0.0):
        raise ValueError(
            f"length and breadth must be positive, got length={length}, breadth={breadth}"
        )
    if not (0.0 <= asymmetry < _MAX_ASYMMETRY):
        raise ValueError(
            f"asymmetry must satisfy 0 <= asymmetry < {_MAX_ASYMMETRY}, got {asymmetry}"
        )

    L = float(length)
    w = float(asymmetry) * L
    t = np.linspace(0.0, np.pi, n)
    z = 0.5 * L * np.cos(t)
    # Poles: cos(0) and cos(pi) give exactly +-L/2, but guard against round-off
    # in the radicand so the radius is exactly zero there.
    radicand = np.clip(L**2 - 4.0 * z**2, 0.0, None)
    r = 0.5 * breadth * np.sqrt(radicand / (L**2 + 8.0 * w * z + 4.0 * w**2))
    r[0] = 0.0
    r[-1] = 0.0
    return r, z


# ---------------------------------------------------------------------------
# gmsh session management
# ---------------------------------------------------------------------------

def _gmsh_initialize_supports_interruptible():
    """True if the installed gmsh accepts ``initialize(interruptible=...)`` (>= 4.12)."""
    try:
        return "interruptible" in inspect.signature(gmsh.initialize).parameters
    except (TypeError, ValueError):  # pragma: no cover - signature unavailable
        return False


@contextlib.contextmanager
def _gmsh_session(model_name):
    """Context manager giving the body a fresh, uniquely named gmsh model.

    * If gmsh is not initialized, it is initialized here (with
      ``interruptible=False`` when the installed gmsh supports it, so the
      builders also work off the main thread) and finalized on exit.
    * If the caller already initialized gmsh, the session is not initialized
      or finalized.  A uniquely named model is added; on exit only that
      model is removed and the caller's current model and
      ``General.Verbosity`` option are restored.

    The body runs with ``General.Verbosity`` set to 0 and the new model as
    the current model, so ``gmsh.write`` writes exactly that model.
    """
    owns_session = not gmsh.isInitialized()
    prev_model = None
    prev_verbosity = None
    if owns_session:
        if _gmsh_initialize_supports_interruptible():
            gmsh.initialize(interruptible=False)
        else:
            gmsh.initialize()
    else:
        prev_verbosity = gmsh.option.getNumber("General.Verbosity")
        prev_model = gmsh.model.getCurrent()
    name = f"{model_name}_{uuid.uuid4().hex[:10]}"
    model_added = False
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add(name)
        model_added = True
        yield name
    finally:
        if owns_session:
            gmsh.finalize()
        else:
            if model_added:
                gmsh.model.setCurrent(name)
                gmsh.model.remove()
            if prev_model in gmsh.model.list():
                gmsh.model.setCurrent(prev_model)
            gmsh.option.setNumber("General.Verbosity", prev_verbosity)


def _gmsh_to_skfem():
    """Write the current gmsh model to a temp file, load it as an skfem mesh, and remove the file."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp_path = f.name
    try:
        gmsh.write(tmp_path)
        # meshio.read prints a stray blank line to stdout; keep library calls silent.
        with contextlib.redirect_stdout(io.StringIO()):
            msh = meshio.read(tmp_path)
        return from_meshio(msh)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Simple circular mesh
# ---------------------------------------------------------------------------

def create_circular_mesh(radius=0.022, mesh_size=0.0015):
    """Create a semicircular mesh (single material region).

    In the axisymmetric formulation this is a sphere of the given radius.

    Parameters
    ----------
    radius : float
        Egg radius [m].
    mesh_size : float
        Characteristic element size [m].

    Returns
    -------
    mesh : skfem.MeshTri
        Mesh with boundaries "shell" and "axis" and the subdomain "egg".
    """
    if not (radius > 0.0 and mesh_size > 0.0):
        raise ValueError(
            f"radius and mesh_size must be positive, got radius={radius}, mesh_size={mesh_size}"
        )
    with _gmsh_session("egg_circle"):
        # Points: center, top, right quadrant, bottom
        p0 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
        p1 = gmsh.model.geo.addPoint(0, radius, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(radius, 0, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(0, -radius, 0, mesh_size)

        # Two quarter-circle arcs (gmsh arcs must be < 180 degrees)
        arc1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
        arc2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
        # Straight line along axis of symmetry
        axis_line = gmsh.model.geo.addLine(p3, p1)

        loop = gmsh.model.geo.addCurveLoop([arc1, arc2, axis_line])
        surf = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        # Physical groups
        gmsh.model.addPhysicalGroup(1, [arc1, arc2], name="shell")
        gmsh.model.addPhysicalGroup(1, [axis_line], name="axis")
        gmsh.model.addPhysicalGroup(2, [surf], name="egg")

        gmsh.model.mesh.generate(2)

        return _gmsh_to_skfem()


# ---------------------------------------------------------------------------
# Realistic egg-shaped mesh with yolk
# ---------------------------------------------------------------------------

def _yolk_clearance(yolk_radius, yolk_center_z, length, breadth, asymmetry):
    """Smallest distance [m] between the yolk circle and the egg outline.

    Positive when the yolk fits inside the shell; computed on a dense
    sampling of the analytic profile.
    """
    r, z = egg_profile(
        n=4001, length=length, breadth=breadth, asymmetry=asymmetry
    )
    return float(np.min(np.hypot(r, z - yolk_center_z)) - yolk_radius)


def create_egg_mesh(
    length=0.055,
    breadth=0.042,
    yolk_radius=0.015,
    yolk_center_z=-0.002,
    asymmetry=0.06,
    mesh_size=0.0015,
    mesh_size_boundary=0.0006,
):
    """Create an egg-shaped mesh with distinct yolk and white regions.

    The outline is the Hugelschaffer profile of :func:`egg_profile` (pointed
    end at +z), represented as a gmsh interpolating (Catmull-Rom) spline
    through 60 profile points.  The yolk is a circle (a sphere in the
    axisymmetric formulation) and the white is the remaining region.

    The default yolk radius of 15 mm gives a yolk of about 28% of the egg
    volume for the default 55 x 42 mm egg, within the 26-31% range of real
    hen eggs.

    Parameters
    ----------
    length, breadth : float
        Egg dimensions [m].
    yolk_radius : float
        Radius of the (circular) yolk region [m].
    yolk_center_z : float
        Axial position of the yolk center [m]; the default is slightly
        toward the blunt end (below the geometric midpoint).
    asymmetry : float
        Egg asymmetry parameter, see :func:`egg_profile`.
    mesh_size : float
        Background element size [m].
    mesh_size_boundary : float
        Element size near the shell boundary [m].

    Returns
    -------
    mesh : skfem.MeshTri
        Mesh with boundaries "shell" and "axis",
        and subdomains "yolk" and "white".

    Raises
    ------
    ValueError
        If any size is not positive, or the yolk does not fit inside the
        shell with a positive white thickness everywhere.
    """
    if not (yolk_radius > 0.0 and mesh_size > 0.0 and mesh_size_boundary > 0.0):
        raise ValueError(
            "yolk_radius, mesh_size and mesh_size_boundary must be positive, got "
            f"{yolk_radius}, {mesh_size}, {mesh_size_boundary}"
        )
    # Validates length, breadth and asymmetry as well.
    clearance = _yolk_clearance(yolk_radius, yolk_center_z, length, breadth, asymmetry)
    if clearance <= 0.0:
        raise ValueError(
            f"yolk (radius {yolk_radius * 1e3:.2f} mm, center z = "
            f"{yolk_center_z * 1e3:.2f} mm) does not fit inside the egg "
            f"({length * 1e3:.1f} x {breadth * 1e3:.1f} mm): the white would "
            f"have thickness {clearance * 1e3:.2f} mm at its thinnest point"
        )

    with _gmsh_session("egg_realistic"):
        # --- Egg outline (right half) as a spline in the (r, z) plane ---
        r_pts, z_pts = egg_profile(
            n=60, length=length, breadth=breadth, asymmetry=asymmetry
        )
        point_ids = []
        for r, z in zip(r_pts, z_pts):
            pid = gmsh.model.geo.addPoint(r, z, 0, mesh_size_boundary)
            point_ids.append(pid)

        # Spline through the profile points (top -> right -> bottom)
        shell_spline = gmsh.model.geo.addSpline(point_ids)
        # point_ids[0] = top of egg on axis, point_ids[-1] = bottom of egg on axis

        # --- Yolk semicircle (right half) ---
        # The yolk axis segment lies on r=0, overlapping the outer axis.
        # To avoid duplicate geometry we split the outer axis into three
        # segments that share endpoints with the yolk boundary.
        yc = yolk_center_z
        yr = yolk_radius

        py0 = gmsh.model.geo.addPoint(0, yc, 0, mesh_size)         # yolk center
        py_top = gmsh.model.geo.addPoint(0, yc + yr, 0, mesh_size)  # top of yolk on axis
        py_mid = gmsh.model.geo.addPoint(yr, yc, 0, mesh_size)      # rightmost yolk point
        py_bot = gmsh.model.geo.addPoint(0, yc - yr, 0, mesh_size)  # bottom of yolk on axis

        # Axis segments (all on r = 0, running bottom -> top)
        axis_below = gmsh.model.geo.addLine(point_ids[-1], py_bot)  # egg bottom -> yolk bottom
        axis_yolk = gmsh.model.geo.addLine(py_bot, py_top)          # yolk bottom -> yolk top
        axis_above = gmsh.model.geo.addLine(py_top, point_ids[0])   # yolk top -> egg top

        # Yolk arcs (right-half semicircle, top -> right -> bottom)
        yarc1 = gmsh.model.geo.addCircleArc(py_top, py0, py_mid)
        yarc2 = gmsh.model.geo.addCircleArc(py_mid, py0, py_bot)

        # Outer loop: shell spline (top -> bottom) + axis segments (bottom -> top)
        outer_loop = gmsh.model.geo.addCurveLoop(
            [shell_spline, axis_below, axis_yolk, axis_above]
        )
        # Yolk loop: arcs (top -> right -> bottom) + the shared axis segment
        # (bottom -> top) form a closed oriented loop.
        yolk_loop = gmsh.model.geo.addCurveLoop([yarc1, yarc2, axis_yolk])

        # Surfaces: white has a yolk hole, yolk fills the hole
        white_surf = gmsh.model.geo.addPlaneSurface([outer_loop, yolk_loop])
        yolk_surf = gmsh.model.geo.addPlaneSurface([yolk_loop])

        gmsh.model.geo.synchronize()

        # Physical groups
        gmsh.model.addPhysicalGroup(1, [shell_spline], name="shell")
        gmsh.model.addPhysicalGroup(
            1, [axis_below, axis_yolk, axis_above], name="axis"
        )
        gmsh.model.addPhysicalGroup(2, [white_surf], name="white")
        gmsh.model.addPhysicalGroup(2, [yolk_surf], name="yolk")

        gmsh.model.mesh.generate(2)

        return _gmsh_to_skfem()


# ---------------------------------------------------------------------------
# Mesh measures
# ---------------------------------------------------------------------------

def mesh_volume(mesh, subdomain=None):
    """Volume [m^3] of the solid of revolution generated by the mesh.

    Computes ``2 pi * integral(r dA)`` over all triangles, which is exact for
    straight-sided triangles (the integral of r over a triangle is its area
    times the r of its centroid).  The result is the volume of the polygonal
    approximation of the domain, slightly below the exact volume of a curved
    outline.

    Parameters
    ----------
    mesh : skfem.MeshTri
        Mesh in the (r, z) half-plane.
    subdomain : str, optional
        Name of a subdomain (e.g. "yolk" or "white") to restrict the
        integral to.  ``None`` (default) integrates over the whole mesh.

    Returns
    -------
    float
        Volume [m^3].
    """
    t = mesh.t
    if subdomain is not None:
        if not mesh.subdomains or subdomain not in mesh.subdomains:
            available = sorted(mesh.subdomains) if mesh.subdomains else []
            raise ValueError(
                f"mesh has no subdomain {subdomain!r} (available: {available})"
            )
        t = t[:, mesh.subdomains[subdomain]]
    r, z = mesh.p[0], mesh.p[1]
    x0, x1, x2 = r[t[0]], r[t[1]], r[t[2]]
    y0, y1, y2 = z[t[0]], z[t[1]], z[t[2]]
    area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    r_centroid = (x0 + x1 + x2) / 3.0
    return float(2.0 * np.pi * np.sum(area * r_centroid))
