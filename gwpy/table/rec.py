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

"""Custom `numpy.recarray` for reading tabular data
"""

import warnings
warnings.warn('The gwpy.table.rec module has been deprecated and will be '
              'removed prior to the 1.0 release.', DeprecationWarning)


from numpy import recarray

from astropy.io import registry as io_registry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class GWRecArray(recarray):
    """`~numpy.recarray` to store tabular data from GW analyses

    See Also
    --------
    numpy.recarray
        for documentation of how to create a new `GWRecArray`
    """

    def __init__(self, *args, **kwargs):
        warnings.warn('The GWRecArray has been deprecated in favour of '
                      'gwpy.table.EventTable, and will be removed prior '
                      'to the 1.0 release', DeprecationWarning)
        return super(GWRecArray, self).__init__(*args, **kwargs)

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `GWRecArray`

        Parameters
        ----------
        source : `str`, `list`, :class:`~glue.lal.Cache`
            source of files, normally a filename or list of filenames,
            or a structured cache of file descriptions

        columns : `list`, optional
            list of column names to read, default reads all available data

        Notes
        -----"""
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this `GWRecArray` to file
        """
        return io_registry.write(cls, source, *args, **kwargs)

    # -- ligolw compatibility -------------------

    def get_column(self, column):
        """Return a column of this array

        This method is provided for compatibility with the
        :class:`glue.ligolw.table.Table`

        Parameters
        ----------
        column : `str`
            name of column (field) to return

        Returns
        -------
        array : `~numpy.ndarray`
            the array for this column
        """
        return self[column]
