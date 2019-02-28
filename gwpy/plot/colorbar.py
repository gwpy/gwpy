# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2019)
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

"""Utilities for generating colour bars for figures
"""

import warnings

from matplotlib import __version__ as mpl_version
from matplotlib.axes import SubplotBase
from matplotlib.colors import LogNorm
from matplotlib.legend import Legend

from .colors import format_norm
from .log import CombinedLogFormatterMathtext

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

LOC_CODES = Legend.codes


# -- custom colorbar generation -----------------------------------------------

def process_colorbar_kwargs(figure, mappable=None, ax=None, use_axesgrid=True,
                            **kwargs):
    # get mappable and axes objects
    if mappable is None and ax is None:
        mappable = find_mappable(*figure.axes)
    if mappable is None:
        mappable = find_mappable(ax)
    if ax is None:
        ax = mappable.axes

    # -- format mappable

    # parse normalisation
    log = kwargs.pop('log', None)
    if log is not None:
        warnings.warn('the `log` keyword to Plot.colorbar is deprecated, '
                      'please pass `norm=\'log\'` in the future to '
                      'request a logarithimic normalisation',
                      DeprecationWarning)
    if log:
        kwargs.setdefault('norm', 'log')

    norm, kwargs = format_norm(kwargs, mappable.norm)
    mappable.set_norm(norm)
    mappable.set_cmap(kwargs.setdefault('cmap', mappable.get_cmap()))

    # -- set tick formatting

    if isinstance(norm, LogNorm):
        kwargs.setdefault('format', CombinedLogFormatterMathtext())

    # -- create axes for colorbar (if required)

    cax = kwargs.pop('cax', None)
    if cax is None:
        # show warning about pending change
        #     current default is to shrink the axes a wee bit,
        #     which will change to no shrinking before 1.0
        if kwargs.get('fraction') is None:
            warnings.warn("`fraction` was not specified. Currently the "
                          "default is `0.15`, matching the previous release "
                          "behaviour, however, this will change to `0.0` in "
                          "an upcoming release. To keep the current default, "
                          "manually specify `fraction=0.15`",
                          PendingDeprecationWarning)
            kwargs['fraction'] = 0.15

        # create axes if requesting not to resize the parent
        if use_axesgrid:
            cax, kwargs = make_axes_axesgrid(ax, **kwargs)
    else:
        kwargs.pop('fraction', None)  # don't need this if already have axes

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


def _get_axes_class(ax):
    if isinstance(ax, SubplotBase):
        return ax._axes_class
    return type(ax)


def _scale_width(value, ax):
    fig = ax.figure
    return value / (ax.get_position().width * fig.get_figwidth())


def _scale_height(value, ax):
    fig = ax.figure
    return value / (ax.get_position().height * fig.get_figheight())


def make_axes_axesgrid(ax, **kwargs):
    kwargs.setdefault('location', 'right')
    if mpl_version >= '1.3.0':
        kwargs.setdefault('ticklocation', kwargs['location'])

    fraction = kwargs.pop('fraction', 0.)
    try:
        if fraction:
            return _make_axes_div(ax, fraction=fraction, **kwargs)
        return _make_axes_inset(ax, **kwargs)
    finally:
        ax.figure.sca(ax)


def _make_axes_div(ax, location='right', fraction=.15, pad=.08, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    axes_class = kwargs.pop('axes_class', _get_axes_class(ax))
    divider = make_axes_locatable(ax)
    return divider.append_axes(location, fraction, pad=pad,
                               axes_class=axes_class), kwargs


def _make_axes_inset(ax, location='right', **kwargs):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    location = location.lower()

    inset_kw = {
        'axes_class': _get_axes_class(ax),
        'bbox_transform': ax.transAxes,
        'borderpad': 0.,
    }

    # get orientation based on location
    if location.lower() in ('left', 'right'):
        pad = kwargs.pop('pad', _scale_width(.1, ax))
        kwargs.setdefault('orientation', 'vertical')
    elif location.lower() in ('top', 'bottom'):
        pad = kwargs.pop('pad', _scale_height(.1, ax))
        kwargs.setdefault('orientation', 'horizontal')
    orientation = kwargs.get('orientation')

    # set params for orientation
    if orientation == 'vertical':
        inset_kw['width'] = .12
        inset_kw['height'] = '100%'
    else:
        inset_kw['width'] = '100%'
        inset_kw['height'] = .12

    # set location and anchor position based on location name
    # NOTE: matplotlib-1.2 requres a location code, and fails on a string
    #       we can move back to just using strings when we drop mpl-1.2
    inset_kw['loc'], inset_kw['bbox_to_anchor'] = {
        'left': (LOC_CODES['lower right'], (-pad, 0., 1., 1.)),
        'right': (LOC_CODES['lower left'], (1+pad, 0., 1., 1.)),
        'bottom': (LOC_CODES['upper left'], (0., -pad,  1., 1.)),
        'top': (LOC_CODES['lower left'], (0., 1+pad, 1., 1.)),
    }[location]

    # allow user overrides
    for key in filter(inset_kw.__contains__, kwargs):
        inset_kw[key] = kwargs.pop(key)

    return inset_axes(ax, **inset_kw), kwargs
