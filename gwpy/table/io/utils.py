# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Utilities for Table I/O
"""

import functools

from astropy.io import registry

from .. import EventTable
from ..filter import filter_table

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def _safe_wraps(wrapper, func):
    try:
        return functools.update_wrapper(wrapper, func)
    except AttributeError:  # func is partial
        return wrapper


def read_with_columns(func):
    """Decorate a Table read method to use the ``columns`` keyword
    """
    def wrapper(*args, **kwargs):
        # parse columns argument
        columns = kwargs.pop("columns", None)

        # read table
        tab = func(*args, **kwargs)

        # filter on columns
        if columns is None:
            return tab
        return tab[columns]

    return _safe_wraps(wrapper, func)


def read_with_selection(func):
    """Decorate a Table read method to apply ``selection`` keyword
    """
    def wrapper(*args, **kwargs):
        """Execute a function, then apply a selection filter
        """
        # parse selection
        selection = kwargs.pop('selection', None) or []

        # read table
        tab = func(*args, **kwargs)

        # apply selection
        if selection:
            return filter_table(tab, selection)

        return tab

    return _safe_wraps(wrapper, func)


# override astropy's readers with decorated versions that accept our
# "selection" keyword argument,
# this is bit hacky, and someone should probably come up with something
# better

def decorate_registered_reader(
        name,
        data_class=EventTable,
        columns=True,
        selection=True,
):
    """Wrap an existing registered reader to use GWpy's input decorators

    Parameters
    ----------
    name : `str`
        the name of the registered format

    data_class : `type`, optional
        the class for whom the format is registered

    columns : `bool`, optional
        use the `read_with_columns` decorator

    selection : `bool`, optional
        use the `read_with_selection` decorator
    """
    reader = registry.get_reader(name, data_class)
    wrapped = (
        read_with_columns(  # use ``columns``
            read_with_selection(  # use ``selection``
                reader
            ),
        )
    )
    return registry.register_reader(name, data_class, wrapped, force=True)


for row in registry.get_formats(data_class=EventTable, readwrite="Read"):
    decorate_registered_reader(row["Format"], data_class=EventTable)
