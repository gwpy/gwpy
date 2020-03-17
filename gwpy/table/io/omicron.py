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

"""Read events from an Omicron-format ROOT file.
"""

from ...io import registry
from .. import EventTable

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def table_from_omicron(source, *args, **kwargs):
    """Read an `EventTable` from an Omicron ROOT file

    This function just redirects to the format='root' reader with appropriate
    defaults.
    """
    if not args:  # only default treename if args not given
        kwargs.setdefault('treename', 'triggers')
    return EventTable.read(source, *args, format='root', **kwargs)


registry.register_reader('root.omicron', EventTable, table_from_omicron)
