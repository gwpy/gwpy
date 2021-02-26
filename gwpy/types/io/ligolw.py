# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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
        """`~xml.sax.handlers.ContentHandler` to read a LIGO_LW ``<Array>``
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
    from ligo.lw.ligolw import Time
    if not isinstance(time, Time):
        t, = time.getElementsByTagName(Time.tagName)
        return _get_time(t)
    return to_gps(time.pcdata)


def _match_time(elem, gps):
    """Returns `True` if the ``elem``'s ``<Time>`` matches ``gps``

    This will return `False` if not exactly one ``<Time>`` element
    is found.
    """
    try:
        return _get_time(elem) == to_gps(gps)
    except (AttributeError, ValueError):  # not exactly one Time
        return False


def _match_array(xmldoc, name=None, epoch=None, **params):
    from ligo.lw.ligolw import Array
    from ligo.lw.param import (Param, get_param)

    def _is_match(arr):
        """Work out whether this `<Array>` element matches the request
        """
        parent = arr.parentNode
        if (
            (name is not None and not _match_name(arr, name))
            or (epoch is not None and not _match_time(parent, epoch))
        ):
            return False
        for key, value in params.items():
            try:
                if get_param(parent, key).pcdata != value:
                    return False
            except ValueError:  # no Param with this Name
                return False
        return True

    def _get_filter_keys(arrays, **given):
        """Returns the set of keyword arguments that can be used to filter

        This is just to format a helpful error message for users to show them
        what params they could use to select the right array.
        """
        # return name and epoch if not given by the user
        keys = set(k for k in ("name", "epoch") if given.pop(k, None) is None)
        # add all of the params found in _any_ array
        return (keys | set(
            p.Name for arr in arrays for
            p in arr.parentNode.getElementsByTagName(Param.tagName)
        )) - set(given.keys())

    # parse out correct element
    matches = list(filter(
        _is_match,
        xmldoc.getElementsByTagName(Array.tagName),
    ))
    try:
        arr, = matches
    except ValueError as exc:
        if not matches:  # no arrays found
            exc.args = ("no <Array> elements found matching request",)
        else:  # multiple arrays found
            keys = _get_filter_keys(matches, name=name, epoch=epoch, **params)
            exc.args = (
                "multiple <Array> elements found matching the request, "
                "please use one of the following keyword arguments to "
                "specify the correct array to read: {}".format(
                    ", ".join(map(repr, sorted(keys))),
                ),
            )
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


def read_series(source, name=None, epoch=None, contenthandler=None, **params):
    """Read a `Series` from ``LIGO_LW`` XML

    Parameters
    ----------
    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`
        file path or open ``LIGO_LW``-format XML file

    name : `str`, optional
        name of the relevant ``<Array>`` element to read

    epoch : `float`, `int`, optional
        GPS time epoch of ``<LIGO_LW>`` element to read

    contenthandler : `~xml.sax.handler.ContentHandler`, optional
        the content handler to use when parsing the document, see
        :func:`series_contenthandler` for the default handler

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
    xmldoc = read_ligolw(
        source,
        contenthandler=contenthandler or series_contenthandler(),
    )
    array = _match_array(xmldoc, name=name, epoch=epoch, **params)

    # parse dimensions
    dims = array.getElementsByTagName(Dim.tagName)
    xdim, ydim = dims
    x0 = xdim.Start
    dx = xdim.Scale
    xunit = xdim.Unit
    if ydim.n > Series._ndim + 1:  # check that we can store these data
        raise ValueError(
            "Cannot parse LIGO_LW Array with {} dimensions in a Series".format(
                ydim.n,
            ),
        )

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
    # note: in order to match the functionality of lal.series,
    #       we discard the actual frequency array, I (DMM) am
    #       not convinced this is the right thing to do, but
    #       it at least provides consistency across multiple
    #       implementations
    return Series(array.array[-1], x0=x0, dx=dx, **array_kw)
