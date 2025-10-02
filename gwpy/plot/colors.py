# Copyright (c) 2017 Louisiana State University
#               2017-2025 Cardiff University
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Colour customisations for visualisation in GWpy."""

from __future__ import annotations

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from typing import TYPE_CHECKING

import numpy
from matplotlib.colors import (
    LogNorm,
    Normalize,
    get_named_colors_mapping,
    hsv_to_rgb,
    rgb_to_hsv,
    to_rgb,
)

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.typing import (
        ColorType,
        RGBColorType,
    )

# -- recommended defaults for current Gravitational-Wave Observatories
# the below colours are designed to work well for the colour-blind, as well
# as in grayscale, so are recommended for publications

GWPY_COLORS = {
    "geo600": "#222222",  # dark gray
    "kagra": "#ffb200",  # yellow/orange
    "ligo-hanford": "#ee0000",  # red
    "ligo-india": "#b0dd8b",  # light green
    "ligo-livingston": "#4ba6ff",  # blue
    "virgo": "#9b59b6",  # magenta/purple
}

# provide user mapping by IFO prefix
_GWO_PREFIX = {
    "geo600": "G1",
    "kagra": "K1",
    "ligo-hanford": "H1",
    "ligo-india": "I1",
    "ligo-livingston": "L1",
    "virgo": "V1",
}
GW_OBSERVATORY_COLORS = {_GWO_PREFIX[n]: GWPY_COLORS[n] for n in GWPY_COLORS}

# set named colour upstream in matplotlib, so users can specify as
# e.g. plot(..., color='gwpy:ligo-hanford')           noqa: ERA001
get_named_colors_mapping().update({
    f"gwpy:{name}": col for name, col in GWPY_COLORS.items()
})


# -- colour utilities ---------------------------------------------------------

def tint(col: ColorType, factor: float = 1.0) -> RGBColorType:
    """Tint a color (make it darker), returning a new RGB array.

    Parameters
    ----------
    col : `matplotlib.typing.ColorType`
        The colour to darken.

    factor : `float`, optional
        The amount by which to darken it. Values less than ``1`` will make things
        darker (``0`` will return black), while less than ``1`` will make things
        lighter.

    Returns
    -------
    tinted : `matplotlib.typing.RGBColorType`
        The new RGB colour tuple.
    """
    # this method is more complicated than it need be to
    # support matplotlib-1.x.
    # for matplotlib-2.x this would just be
    #     h, s, v = rgb_to_hsv(to_rgb(c))  noqa: ERA001
    #     v *= factor                      noqa: ERA001
    #     return hsv_to_rgb((h, s, v))     noqa: ERA001
    rgb = numpy.array(to_rgb(col), ndmin=3)
    hsv = rgb_to_hsv(rgb)
    hsv[-1][-1][2] *= factor
    return hsv_to_rgb(hsv)[-1][-1]


def format_norm(
    kwargs: dict[str, Any],
    current: str | Normalize | None = None,
) -> tuple[Normalize, dict[str, Any]]:
    """Format a `~matplotlib.colors.Normalize` from a set of kwargs.

    Parameters
    ----------
    kwargs : `dict`
        A set of keyword arguments to a plotting routine.

    current : `Normalize`, optional
        The current normalization, which will be preserved if
        ``norm`` is not specified in ``kwargs``.

    Returns
    -------
    norm, kwargs
        The formatted `Normalize` instance, and the remaining keywords.

    Raises
    ------
    ValueError
        If ``"norm"`` is included in ``kwargs`` and doesn't match a
        recognised value (one of ``"linear"``, ``"log"`` or a
        `~matplotlib.colors.Normalize` instance).
    """
    norm = kwargs.pop("norm", current) or "linear"
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    clim = kwargs.pop("clim", (vmin, vmax)) or (None, None)
    clip = kwargs.pop("clip", None)

    if norm == "linear":
        norm = Normalize()
    elif norm == "log":
        norm = LogNorm()
    elif not isinstance(norm, Normalize):
        msg = f"unrecognised value for norm '{norm}'"
        raise ValueError(msg)

    for attr, value in (
        ("vmin", clim[0]),
        ("vmax", clim[1]),
        ("clip", clip),
    ):
        if value is not None:
            setattr(norm, attr, value)

    return norm, kwargs
