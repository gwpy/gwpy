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

"""Read/write series in LIGO_LW-XML
"""

from .. import Series
from ...io.ligolw import read_ligolw
from ...time import to_gps

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def series_contenthandler():
    """Build a `~xml.sax.handlers.ContentHandler` to read a LIGO_LW <Array>
    """
    from ligo.lw import (
        ligolw,
        array as ligolw_array,
        param as ligolw_param
    )

    @ligolw_array.use_in
    @ligolw_param.use_in
    class ArrayContentHandler(ligolw.LIGOLWContentHandler):
        """`~xml.sax.handlers.ContentHandler` to read a LIGO_LW <Array>
        """
        pass

    return ArrayContentHandler


def _match_name(elem, name):
    """Returns `True` if the ``elem``'s Name matches ``name``
    """
    try:
        return elem.Name == name
    except AttributeError:  # no name
        return False


def _get_time(time):
    """Returns the Time element of a ``<LIGO_LW>``.
    """
    return to_gps(time.pcdata)


def _match_time(elem, gps):
    """Returns `True` if the ``elem``'s ``<Time>`` matches ``gps``

    This will return `False` if not exactly one ``<Time>`` element
    is found.
    """
    from ligo.lw.ligolw import Time
    try:
        time, = elem.getElementsByTagName(Time.tagName)
    except ValueError:  # multiple times
        return False
    return _get_time(time) == to_gps(gps)


def _match_array(xmldoc, name=None, epoch=None, **params):
    from ligo.lw.ligolw import Array
    from ligo.lw.param import get_param

    def _is_match(arr):
        parent = arr.parentNode
        if (
            (name is not None and not _match_name(arr, name)) or
            (epoch is not None and not _match_time(parent, epoch))
        ):
            return False
        for key, value in params.items():
            try:
                if get_param(parent, key).pcdata != value:
                    return False
            except ValueError:  # no Param with this Name
                return False
        return True

    # parse out correct element
    matches = filter(_is_match, xmldoc.getElementsByTagName(Array.tagName))
    try:
        arr, = matches
    except ValueError as exc:
        if not list(matches):
            exc.args = ("no <Array> elements found matching request",)
        else:
            exc.args = ("multiple <Array> elements found matching request, "
                        "consider using the `name`, passing `epoch` or param "
                        "`<name>=<value>` kwargs to filter the correct parent "
                        "<LIGO_LW> element",)
        raise

    return arr


def _update_metadata_from_ligolw(array, kwargs):
    from ligo.lw.ligolw import Time
    from ligo.lw.param import get_param

    parent = array.parentNode

    # pick out the epoch
    try:
        time, = parent.getElementsByTagName(Time.tagName)
    except ValueError:
        pass
    else:
        kwargs.setdefault('epoch', _get_time(time))

    # copy over certain other params, if they exist
    for key in ('channel',):
        try:
            kwargs[key] = get_param(parent, key)
        except ValueError:
            pass

    return kwargs


def read_series(source, name=None, epoch=None, **params):
    """Read a `Series` from ``LIGO_LW`` XML

    Parameters
    ----------
    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`
        file path or open ``LIGO_LW``-format XML file

    name : `str`, optional
        name of the relevant ``<Array>`` element to read

    epoch : `float`, `int`, optional
        GPS time epoch of ``<LIGO_LW>`` element to read

    **params
        other ``<Param>`` ``(name, value)`` pairs to use in matching
        the parent correct ``<LIGO_LW>`` element to read

    Returns
    -------
    series : `~gwpy.types.Series`
        a series with metadata read from the ``<Array>``
    """
    from ligo.lw.ligolw import Dim

    # read document and find relevant <Array> element
    xmldoc = read_ligolw(source, contenthandler=series_contenthandler())
    array = _match_array(xmldoc, name=name, epoch=epoch, **params)

    # parse dimensions
    dims = array.getElementsByTagName(Dim.tagName)
    xdim, ydim = dims
    x0 = xdim.Start
    dx = xdim.Scale
    xunit = xdim.Unit
    if ydim.n > Series._ndim + 1:
        raise ValueError("Cannot parse LIGO_LW Array with {} "
                         "dimensions".format(ydim.n))

    # parse metadata
    array_kw = {
        'name': array.Name,
        'unit': array.Unit,
        'xunit': xunit,
    }

    # update metadata from parent <LIGO_LW> element
    array_kw = _update_metadata_from_ligolw(array, array_kw)

    # normalize units (mainly for FrequencySeries)
    if array_kw.get("xunit") == "s^-1":
        array_kw["xunit"] = "Hz"
    if (array_kw.get("unit") or "").startswith("s "):
        array_kw["unit"] = array_kw["unit"].split(" ", 1)[1] + " Hz^-1"

    # build Series
    try:
        xindex, value = array.array
    except ValueError:  # not two dimensions stored
        return Series(array.array[0], x0=x0, dx=dx, **array_kw)
    return Series(value, xindex=xindex, **array_kw)
