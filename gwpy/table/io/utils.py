# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

from functools import wraps

from ..filter import filter_table

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def read_with_selection(func):
    """Decorate a Table read method to apply ``selection`` keyword
    """
    @wraps(func)
    def decorated_func(*args, **kwargs):
        """Execute a function, then apply a selection filter
        """
        # parse selection
        try:
            selection = kwargs.pop('selection')
        except KeyError:
            selection = []

        # read table
        tab = func(*args, **kwargs)

        # apply selection
        if selection:
            return filter_table(tab, selection)

        return tab

    return decorated_func
