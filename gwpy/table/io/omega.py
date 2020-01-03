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

"""Read events from an Omega-format ASCII file.
"""

import re

from astropy.io.ascii import core

from ...io import registry
from .. import (Table, EventTable)
from .utils import decorate_registered_reader


class OmegaHeader(core.BaseHeader):
    """Parser for Omega ASCII header
    """
    def get_cols(self, lines):
        """Initialize Column objects from a multi-line ASCII header

        Parameters
        ----------
        lines : `list`
            List of table lines
        """
        re_name_def = re.compile(r'^\s*%\s+(?P<colname>\w+)')
        self.names = []
        for line in lines:
            if not line:  # ignore empty lines in header (windows)
                continue
            if not line.startswith('%'):  # end of header lines
                break
            match = re_name_def.search(line)
            if match:
                self.names.append(match.group('colname'))

        if not self.names:
            raise core.InconsistentTableError(
                'No column names found in Omega header')

        self.cols = []  # pylint: disable=attribute-defined-outside-init
        for name in self.names:
            col = core.Column(name=name)
            self.cols.append(col)

    def write(self, lines):
        for name in self.colnames:
            lines.append('%% %s' % name)


class OmegaData(core.BaseData):
    """Parser for Omega ASCII data
    """
    comment = '%'


class Omega(core.BaseReader):
    """Read an Omega file
    """
    _format_name = 'omega'
    _io_registry_can_write = True
    _description = 'Omega format table'

    header_class = OmegaHeader
    data_class = OmegaData


# register for EventTable
# (to populate that object's docstring)
registry.register_reader(
    'ascii.omega',
    EventTable,
    registry.get_reader("ascii.omega", Table),
)
decorate_registered_reader(
    "ascii.omega",
    EventTable,
)
