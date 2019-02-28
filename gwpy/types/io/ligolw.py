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
    from glue.ligolw import (
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


def read_series(source, name, match=None):
    """Read a `Series` from LIGO_LW-XML

    Parameters
    ----------
    source : `file`, `str`, :class:`~glue.ligolw.ligolw.Document`
        file path or open LIGO_LW-format XML file

    name : `str`
        name of the relevant `LIGO_LW` element to read

    match : `dict`, optional
        dict of (key, value) `Param` pairs to match correct LIGO_LW element,
        this is useful if a single file contains multiple `LIGO_LW` elements
        with the same name
    """
    from glue.ligolw.ligolw import (LIGO_LW, Time, Array, Dim)
    from glue.ligolw.param import get_param

    # read document
    xmldoc = read_ligolw(source, contenthandler=series_contenthandler())

    # parse match dict
    if match is None:
        match = dict()

    def _is_match(elem):
        try:
            if elem.Name != name:
                return False
        except AttributeError:  # Name is not set
            return False
        for key, value in match.items():
            try:
                if get_param(elem, key).pcdata != value:
                    return False
            except ValueError:  # no Param with this Name
                return False
        return True

    # parse out correct element
    matches = filter(_is_match, xmldoc.getElementsByTagName(LIGO_LW.tagName))
    try:
        elem, = matches
    except ValueError as exc:
        if not matches:
            exc.args = ("no LIGO_LW elements found matching request",)
        else:
            exc.args = ('multiple LIGO_LW elements found matching request, '
                        'please consider using `match=` to select the '
                        'correct element',)
        raise

    # get data
    array, = elem.getElementsByTagName(Array.tagName)

    # parse dimensions
    dims = array.getElementsByTagName(Dim.tagName)
    xdim = dims[0]
    x0 = xdim.Start
    dx = xdim.Scale
    xunit = xdim.Unit
    try:
        ndim = dims[1].n
    except IndexError:
        pass
    else:
        if ndim > 2:
            raise ValueError("Cannot parse LIGO_LW Array with {} "
                             "dimensions".format(ndim))

    # parse metadata
    array_kw = {
        'name': array.Name,
        'unit': array.Unit,
        'xunit': xunit,
    }
    try:
        array_kw['epoch'] = to_gps(
            elem.getElementsByTagName(Time.tagName)[0].pcdata)
    except IndexError:
        pass
    for key in ('channel',):
        try:
            array_kw[key] = get_param(elem, key)
        except ValueError:
            pass

    # build Series
    try:
        xindex, value = array.array
    except ValueError:  # not two dimensions stored
        return Series(array.array[0], x0=x0, dx=dx, **array_kw)
    return Series(value, xindex=xindex, **array_kw)
