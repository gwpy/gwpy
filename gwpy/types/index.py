# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2016)
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

"""Quantity array for indexing a Series
"""

import numpy

from astropy.units import Quantity


class Index(Quantity):
    """1-D `~astropy.units.Quantity` array for indexing a `Series`
    """
    @property
    def regular(self):
        """`True` if this index is linearly increasing
        """
        try:
            return self.info.meta['regular']
        except (TypeError, KeyError):
            if self.info.meta is None:
                self.info.meta = {}
            self.info.meta['regular'] = self.is_regular()
            return self.info.meta['regular']

    def is_regular(self):
        """Determine whether this `Index` contains linearly increasing samples

        This also works for linear decrease
        """
        if self.size <= 1:
            return False
        return numpy.isclose(numpy.diff(self.value, n=2), 0).all()

    def __getitem__(self, key):
        item = super(Index, self).__getitem__(key)
        if item.isscalar:
            return item.view(Quantity)
        return item
