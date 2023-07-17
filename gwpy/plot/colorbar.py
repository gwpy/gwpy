# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2017)
#               Cardiff University (2017-2023)
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

"""Utilities for generating colour bars for figures.

This module mainly exists to support generating colour bars for figures
where the 'parent' Axes isn't resized to accommodate the new Axes for
the colour bar.

GWpy adds this functionality by overloading the `Figure.colorbar` method
with calls to functions below that add support for handling ``fraction=0.``.
"""

from matplotlib.colors import LogNorm

from .colors import format_norm
from .log import LogFormatter

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- custom colorbar generation -----------------------------------------------

def process_colorbar_kwargs(
    figure,
    mappable=None,
    ax=None,
    cax=None,
    fraction=0.,
    **kwargs,
):
    """Internal function to configure the keyword arguments for colorbars.

    The main purpose of this function is to replace the default matplotlib
    behaviour (resizing the 'parent' axes to make space for the colorbar
    axes) with our default or creating a new axes alongside the parent
    axes without resizing.

    Parameters
    ----------
    figure : `matplotlib.figure.Figure`
        The figure on which to draw the new colorbar axes.

    mappable : matplotlib data collection
        Collection against which to map the colouring, default will
        be the last added mappable artist (collection or image)

    ax : `matplotlib.axes.Axes`
        The `Axes` against which to anchor the colorbar Axes.

    cax : `matplotlib.axes.Axes`
        The `Axes` on which to draw the colorbar.

    fraction : `float`
        The fraction of space to steal from the parent Axes to make
        space for the colourbar.

    **kwargs
        Other keyword arguments to pass to
        :meth:`matplotlib.figure.Figure.colorbar`

    Returns
    -------
    mappable
        The Collection against which to map the colouring.

    kwargs
        A dict of keyword arguments to pass to
        :meth:`matplotlib.figure.Figure.colorbar`.
    """
    # get mappable and axes objects
    if mappable is None and ax is None:
        mappable = find_mappable(*figure.axes)
    if mappable is None:
        mappable = find_mappable(ax)
    if ax is None:
        ax = mappable.axes

    # -- format mappable

    # parse normalisation
    norm, kwargs = format_norm(kwargs, mappable.norm)
    mappable.set_norm(norm)
    mappable.set_cmap(kwargs.pop('cmap', mappable.get_cmap()))

    # -- set tick formatting

    if isinstance(norm, LogNorm):
        kwargs.setdefault('format', LogFormatter())

    # -- create axes for colorbar (if required)

    if cax is not None:  # cax was given, we don't need fraction
        kwargs.pop("fraction", None)
    elif fraction == 0.:  # if fraction is 0, make the inset axes ourselves
        cax, kwargs = _make_inset_axes(ax, **kwargs)
    else:  # otherwise let matplotlib generate the Axes using its own default
        kwargs["fraction"] = fraction

    # pack kwargs
    kwargs.update(ax=ax, cax=cax)
    return mappable, kwargs


# -- utilities ----------------------------------------------------------------

def find_mappable(*axes):
    """Find the most recently added mappable layer in the given axes

    Parameters
    ----------
    *axes : `~matplotlib.axes.Axes`
        one or more axes to search for a mappable
    """
    for ax in axes:
        for aset in ('collections', 'images'):
            try:
                return getattr(ax, aset)[-1]
            except (AttributeError, IndexError):
                continue
    raise ValueError("Cannot determine mappable layer on any axes "
                     "for this colorbar")


def _scale_width(value, ax):
    fig = ax.figure
    return value / (ax.get_position().width * fig.get_figwidth())


def _scale_height(value, ax):
    fig = ax.figure
    return value / (ax.get_position().height * fig.get_figheight())


def _colorbar_bounds(
    ax,
    location="right",
    width=None,
    length=1.,
    orientation=None,
    pad=None,
):
    """Returns the ``bounds`` for an inset Axes designed for a colourbar.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes to anchor to.

    location : `str`
        Where to place the colourbar, one of

        - ``"left"``
        - ``"right"`` (default)
        - ``"top"``
        - ``"bottom"``

    width : `float`
        The size of the colourbar along its short axis (i.e. 'width' for
        a vertical bar, 'height' for a horizontal bar), as a fraction of
        the parent `Axes` size in the same direction.

    length : `float`
        The size of the colourbar along its long axis (i.e. 'height' for
        a vertical bar, 'length' for a horizontal bar), as a fraction of
        the parent `Axes` size in the same direction.

    orientation : `str` or `None`
        One of ``"horizontal"`` or ``"vertical"``. The default is
        ``"horizontal"`` if ``location`` is ``"top"`` or ``"bottom"``,
        otherwise ``"vertical"``.

    pad : `float`
        The gap between the parent Axes and the colourbar, as a fraction
        of the parent Axes length in the relevant direction. Default is
        equivalent to .1 inches.

    Returns
    -------
    bounds : `tuple` of `float`
        The ``(x0, y0, width, height)`` bounds for an inset Axes to
        contain a colourbar. This is designed to be passed to
        :meth:`~matplotlib.axes.Axes.inset_axes`.
    """
    location = location.lower()

    # calculate default width and padding for the relevant orientation
    orientation = "vertical" if location in ('left', 'right') else "horizontal"
    if orientation == "vertical":
        size = _scale_width(.1, ax)
    elif orientation == "horizontal":
        size = _scale_height(.1, ax)
    pad = size if pad is None else pad
    width = size if width is None else width

    # where to start on the long axis (for centre-aligned Axes)
    l0 = .5 - length / 2.

    if location == "left":
        return 0 - pad - width, l0, width, length
    if location == "right":
        return 1 + pad, l0, width, length
    if location == "top":
        return l0, 1 + pad, length, width
    # "bottom":
    return l0, 0 - pad - width, length, width


def _make_inset_axes(
    ax,
    location='right',
    width=0.012,
    length=1.,
    pad=None,
    **kwargs,
):
    """Create a new `Axes` to support a colorbar using `Axes.inset_axes`.
    """
    # set default orientation
    if location in ('left', 'right'):
        orientation = kwargs.setdefault("orientation", "vertical")
    elif location in ('top', 'bottom'):
        orientation = kwargs.setdefault("orientation", "horizontal")
    else:
        raise ValueError(f"inset Axes location '{location}' not recognised")

    # set default ticklocation for the right location/orientation
    if location == "top" and orientation == "horizontal":
        kwargs.setdefault("ticklocation", "top")
    if location == "left" and orientation == "vertical":
        kwargs.setdefault("ticklocation", "left")

    # calculate the bounds for the new Axes
    bounds = _colorbar_bounds(
        ax,
        location=location,
        width=width,
        length=length,
        orientation=orientation,
        pad=pad,
    )

    return ax.inset_axes(bounds, transform=ax.transAxes), kwargs
