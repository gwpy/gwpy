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

"""Read events from Coherent Wave-Burst (cWB)-format ROOT files.
"""

import re

from astropy.io.ascii import core

from ...io import registry
from .. import (Table, EventTable)
from .utils import decorate_registered_reader

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- ROOT ---------------------------------------------------------------------

def table_from_cwb(source, *args, **kwargs):
    """Read an `EventTable` from a Coherent WaveBurst ROOT file

    This function just redirects to the format='root' reader with appropriate
    defaults.
    """
    return EventTable.read(source, 'waveburst', *args, format='root', **kwargs)


registry.register_reader('root.cwb', EventTable, table_from_cwb)


# -- ASCII --------------------------------------------------------------------

class CwbHeader(core.BaseHeader):
    """Parser for cWB ASCII header
    """

    def get_cols(self, lines):
        """Initialize Column objects from a multi-line ASCII header

        Parameters
        ----------
        lines : `list`
            List of table lines
        """
        re_name_def = re.compile(
            r'^\s*#\s+'  # whitespace and comment marker
            r'(?P<colnumber>[0-9]+)\s+-\s+'  # number of column
            r'(?P<colname>(.*))'
        )
        self.names = []
        include_cuts = False
        for line in lines:
            if not line:  # ignore empty lines in header (windows)
                continue
            if not line.startswith('# '):  # end of header lines
                break
            if line.startswith('# -/+'):
                include_cuts = True
            else:
                match = re_name_def.search(line)
                if match:
                    self.names.append(match.group('colname').rstrip())

        if not self.names:
            raise core.InconsistentTableError(
                'No column names found in cWB header')

        if include_cuts:
            self.cols = [  # pylint: disable=attribute-defined-outside-init
                core.Column(name='selection cut 1'),
                core.Column(name='selection cut 2'),
            ]
        else:
            self.cols = []  # pylint: disable=attribute-defined-outside-init
        for name in self.names:
            col = core.Column(name=name)
            self.cols.append(col)

    def write(self, lines):
        if 'selection cut 1' in self.colnames:
            lines.append('# -/+ - not passed/passed final selection cuts')
        for i, name in enumerate(self.colnames):
            lines.append('# %.2d - %s' % (i+1, name))


class CwbData(core.BaseData):
    """Parser for cWB ASCII data
    """
    comment = '#'


class Cwb(core.BaseReader):
    """Read an Cwb file
    """
    _format_name = 'cwb'
    _io_registry_can_write = True
    _description = 'cWB EVENTS format table'

    header_class = CwbHeader
    data_class = CwbData


# register for EventTable
registry.register_reader(
    "ascii.cwb",
    EventTable,
    registry.get_reader("ascii.cwb", Table),
)
decorate_registered_reader(
    "ascii.cwb",
    EventTable,
)
