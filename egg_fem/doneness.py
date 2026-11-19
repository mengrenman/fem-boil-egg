"""Doneness criteria for boiled eggs, evaluated on a finished HeatSolver run.

Definitions
-----------
Three doneness levels are defined from the temperature field (not from the
temperature at one fixed probe location):

``soft``    (soft-boiled: white set, yolk runny)
    The coldest point of the white reaches 63 °C (``WHITE_DENATURATION_K``).
    Egg white starts to set at about 63 °C (ovotransferrin, the first major
    white protein, denatures near 62-63 °C).  The coldest white point lies on
    the white-yolk interface (for the default egg, on the axis at the top of
    the yolk), so at this moment all of the white has reached its set
    temperature and even the coolest point of the yolk surface is at 63 °C.
    This is the criterion of C. D. H. Williams' egg-timing formula: the egg
    "will be ready when the temperature at the boundary of the yolk has
    risen to T_yolk ~ 63 °C" (C. D. H. Williams, "The science of boiling an
    egg", University of Exeter,
    https://www.exeter.ac.uk/research-centres/theoretical-physics/boiling-an-egg/).
    Williams applies it to a homogeneous sphere; here it is applied to the
    coldest point of the yolk boundary of the two-region egg.

    The criterion is a single-point one: the coldest white point is the top
    or the bottom of the yolk on the symmetry axis, so the soft time moves
    with the axial position of the yolk (see the yolk-offset scan in
    notebook 03: roughly 25 s per mm, with a kink where the coldest point
    switches from the top to the bottom of the yolk), while the medium and
    hard times, which depend on the yolk as a whole, move by a few seconds.

``medium``  (medium / jammy: outer yolk set, center still soft)
    At least 50% of the yolk volume is at or above 70 °C
    (``YOLK_FULL_SET_K``; the yolk sets at about 70 °C).  The fraction is the
    true (axisymmetric, ``r``-weighted) volume fraction.

``hard``    (hard-boiled)
    The coldest point of the yolk (for the default egg, its center) reaches
    70 °C.

As secondary diagnostics, ``center_crossing_times`` returns the times at
which the yolk center crosses 63, 65 and 70 °C.

These are modeling choices motivated by protein denaturation temperatures,
not measured doneness data.  Agreement of the resulting times with cooking
guides is a consistency check, not an independent validation of the model.
"""

import numpy as np

from .materials import WHITE_DENATURATION_K, YOLK_FULL_SET_K

# Fraction of the yolk volume that must be at or above YOLK_FULL_SET_K for
# the "medium" (jammy) doneness level.
MEDIUM_YOLK_FRACTION = 0.5

# Human-readable definitions, e.g. for figure legends and notebook tables.
DONENESS_DEFINITIONS = {
    "soft": "coldest point of the white reaches 63 °C (white set, yolk runny)",
    "medium": "at least 50% of the yolk volume is at or above 70 °C (outer yolk set, center soft)",
    "hard": "coldest point of the yolk reaches 70 °C (yolk set throughout)",
}

_KELVIN = 273.15


def crossing_time(times, values, threshold):
    """First time at which ``values`` rises to ``threshold``.

    The crossing is located by linear interpolation between the two samples
    that bracket it.

    Parameters
    ----------
    times, values : array_like
        Sample times and values of equal length.
    threshold : float
        Threshold in the units of ``values``.

    Returns
    -------
    float
        The first upward crossing time, ``times[0]`` if the first sample is
        already at or above the threshold, and ``nan`` if the threshold is
        never reached.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of equal length")
    above = np.nonzero(values >= threshold)[0]
    if len(above) == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return float(times[0])
    v0, v1 = values[i - 1], values[i]
    return float(times[i - 1] + (threshold - v0) * (times[i] - times[i - 1]) / (v1 - v0))


def doneness_times(
    solver,
    white_set_K=WHITE_DENATURATION_K,
    yolk_set_K=YOLK_FULL_SET_K,
    medium_fraction=MEDIUM_YOLK_FRACTION,
):
    """Times [s] at which the egg reaches soft, medium and hard doneness.

    See the module docstring for the definitions.  The solver must already
    have been run far enough; a level that is not reached within the
    simulated time is reported as ``nan``.

    Parameters
    ----------
    solver : HeatSolver
        A solver on a mesh with "white" and "yolk" subdomains.
    white_set_K : float
        White set temperature [K] (default 63 °C).
    yolk_set_K : float
        Yolk set temperature [K] (default 70 °C).
    medium_fraction : float
        Yolk volume fraction at or above ``yolk_set_K`` that defines medium
        doneness (default 0.5).

    Returns
    -------
    dict
        ``{"soft": t, "medium": t, "hard": t}`` in seconds.

    Raises
    ------
    ValueError
        If the mesh does not have both a "white" and a "yolk" subdomain.
    """
    subdomains = solver.mesh.subdomains or {}
    missing = [name for name in ("white", "yolk") if name not in subdomains]
    if missing:
        raise ValueError(
            "doneness_times needs a mesh with 'white' and 'yolk' subdomains; "
            f"missing: {missing}"
        )

    times, white_min_C = solver.region_min_temperature_history("white")
    _, yolk_min_C = solver.region_min_temperature_history("yolk")
    _, yolk_fraction = solver.region_fraction_above_history("yolk", yolk_set_K)

    return {
        "soft": crossing_time(times, white_min_C, white_set_K - _KELVIN),
        "medium": crossing_time(times, yolk_fraction, medium_fraction),
        "hard": crossing_time(times, yolk_min_C, yolk_set_K - _KELVIN),
    }


def center_crossing_times(solver, thresholds_C=(63.0, 65.0, 70.0)):
    """Times [s] at which the yolk center (the egg center) crosses temperatures.

    Uses the P1-interpolated temperature at ``solver.center_point()``.

    Parameters
    ----------
    solver : HeatSolver
    thresholds_C : sequence of float
        Temperatures [°C].

    Returns
    -------
    dict
        ``{threshold_C: time_s}`` with float keys; ``nan`` for a threshold
        that is not reached within the simulated time.
    """
    times, T_center_C = solver.center_temperature_history()
    return {
        float(thr): crossing_time(times, T_center_C, float(thr))
        for thr in thresholds_C
    }
