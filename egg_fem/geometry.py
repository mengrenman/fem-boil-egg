"""Mesh generation for axisymmetric egg models.

Creates 2D meshes in the (r, z) half-plane for use with an axisymmetric
formulation.  The axis of symmetry is r = 0 (the left edge).

Two mesh generators are provided:
  - create_circular_mesh:  simple semicircular cross-section (single region)
  - create_egg_mesh:       realistic egg shape with separate yolk region

Both return a scikit-fem MeshTri with named boundaries and subdomains.
"""

import tempfile
import numpy as np
import gmsh
import meshio
from skfem import MeshTri
from skfem.io import from_meshio


# ---------------------------------------------------------------------------
# Egg profile parameterisation
# ---------------------------------------------------------------------------

def egg_profile(n=200, length=0.055, breadth=0.042):
    """Return (r, z) points tracing the right-half egg profile.

    The egg is centred at the origin with its long axis along z.
    Only the right half (r >= 0) is returned for axisymmetric use.

    Parameters
    ----------
    n : int
        Number of points on the profile.
    length : float
        Egg length pole-to-pole [m].
    breadth : float
        Maximum egg diameter [m].

    Returns
    -------
    r, z : ndarray, shape (n,)
    """
    t = np.linspace(0, np.pi, n)
    a = breadth / 2
    b = length / 2
    r = a * np.sin(t)
    # slight asymmetry: pointed top (positive z)
    z = b * np.cos(t) * (1.0 + 0.15 * np.sin(t))
    return r, z


# ---------------------------------------------------------------------------
# Simple circular mesh
# ---------------------------------------------------------------------------

def create_circular_mesh(radius=0.022, mesh_size=0.0015):
    """Create a semicircular mesh (single material region).

    Parameters
    ----------
    radius : float
        Egg radius [m].
    mesh_size : float
        Characteristic element size [m].

    Returns
    -------
    mesh : skfem.MeshTri
        Mesh with boundaries "shell" and "axis".
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("egg_circle")

        # Points: centre, top, right quadrant, bottom
        p0 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
        p1 = gmsh.model.geo.addPoint(0, radius, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(radius, 0, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(0, -radius, 0, mesh_size)

        # Two quarter-circle arcs (gmsh arcs must be < 180°)
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

        mesh = _gmsh_to_skfem()
    finally:
        gmsh.finalize()
    return mesh


# ---------------------------------------------------------------------------
# Realistic egg-shaped mesh with yolk
# ---------------------------------------------------------------------------

def create_egg_mesh(
    length=0.055,
    breadth=0.042,
    yolk_radius=0.010,
    yolk_center_z=-0.002,
    mesh_size=0.0015,
    mesh_size_boundary=0.0006,
):
    """Create a realistic egg mesh with distinct yolk and white regions.

    Parameters
    ----------
    length, breadth : float
        Egg dimensions [m].
    yolk_radius : float
        Radius of the (circular) yolk region [m].
    yolk_center_z : float
        Vertical offset of yolk centre [m] (slightly below geometric centre).
    mesh_size : float
        Background element size [m].
    mesh_size_boundary : float
        Element size near the shell boundary [m].

    Returns
    -------
    mesh : skfem.MeshTri
        Mesh with boundaries "shell" and "axis",
        and subdomains "yolk" and "white".
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("egg_realistic")

        # --- Egg outline (right half) as a spline in the (r, z) plane ---
        r_pts, z_pts = egg_profile(n=60, length=length, breadth=breadth)
        point_ids = []
        for r, z in zip(r_pts, z_pts):
            ms = mesh_size_boundary if r < 0.002 else mesh_size
            pid = gmsh.model.geo.addPoint(r, z, 0, ms)
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

        py0 = gmsh.model.geo.addPoint(0, yc, 0, mesh_size)        # yolk centre
        py_top = gmsh.model.geo.addPoint(0, yc + yr, 0, mesh_size) # top of yolk on axis
        py_mid = gmsh.model.geo.addPoint(yr, yc, 0, mesh_size)     # rightmost yolk point
        py_bot = gmsh.model.geo.addPoint(0, yc - yr, 0, mesh_size) # bottom of yolk on axis

        # Axis segments (all on r = 0, running bottom → top)
        axis_below = gmsh.model.geo.addLine(point_ids[-1], py_bot)   # egg bottom → yolk bottom
        axis_yolk  = gmsh.model.geo.addLine(py_bot, py_top)          # yolk bottom → yolk top
        axis_above = gmsh.model.geo.addLine(py_top, point_ids[0])    # yolk top → egg top

        # Yolk arcs (right-half semicircle, top → right → bottom)
        yarc1 = gmsh.model.geo.addCircleArc(py_top, py0, py_mid)
        yarc2 = gmsh.model.geo.addCircleArc(py_mid, py0, py_bot)

        # Loops
        # Outer loop: shell spline (top→bottom) + axis segments (bottom→top)
        outer_loop = gmsh.model.geo.addCurveLoop(
            [shell_spline, axis_below, axis_yolk, axis_above]
        )
        # Yolk loop: arcs (top→right→bottom) + shared axis reversed (bottom→top)
        # CurveLoop expects a closed oriented loop. The arcs go top→bot,
        # and axis_yolk goes bot→top, so we negate it to get top→bot direction…
        # Actually: yarc1(top→mid) + yarc2(mid→bot) + axis_yolk(bot→top) = closed ✓
        yolk_loop = gmsh.model.geo.addCurveLoop([yarc1, yarc2, axis_yolk])

        # Surfaces: white has yolk hole, yolk fills the hole
        white_surf = gmsh.model.geo.addPlaneSurface([outer_loop, yolk_loop])
        yolk_surf  = gmsh.model.geo.addPlaneSurface([yolk_loop])

        gmsh.model.geo.synchronize()

        # Physical groups
        gmsh.model.addPhysicalGroup(1, [shell_spline], name="shell")
        gmsh.model.addPhysicalGroup(
            1, [axis_below, axis_yolk, axis_above], name="axis"
        )
        gmsh.model.addPhysicalGroup(2, [white_surf], name="white")
        gmsh.model.addPhysicalGroup(2, [yolk_surf], name="yolk")

        gmsh.model.mesh.generate(2)

        mesh = _gmsh_to_skfem()
    finally:
        gmsh.finalize()
    return mesh


# ---------------------------------------------------------------------------
# Helper: gmsh -> skfem via meshio
# ---------------------------------------------------------------------------

def _gmsh_to_skfem():
    """Write the current gmsh model to a temp file, load as skfem mesh, and remove temp file."""
    import os
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp_path = f.name
    try:
        gmsh.write(tmp_path)
        msh = meshio.read(tmp_path)
        return from_meshio(msh)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
