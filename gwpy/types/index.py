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

"""Quantity array for indexing a Series
"""

import numpy

from astropy.units import Quantity


class Index(Quantity):
    """1-D `~astropy.units.Quantity` array for indexing a `Series`
    """
    @classmethod
    def define(cls, start, step, num, dtype=None):
        """Define a new `Index`.

        The output is basically::

            start + numpy.arange(num) * step

        Parameters
        ----------
        start : `Number`
            The starting value of the index.

        step : `Number`
            The step size of the index.

        num : `int`
            The size of the index (number of samples).

        dtype : `numpy.dtype`, `None`, optional
            The desired dtype of the index, if not given, defaults
            to the higher-precision dtype from ``start`` and ``step``.

        Returns
        -------
        index : `Index`
            A new `Index` created from the given parameters.
        """
        if dtype is None:
            dtype = max(
                numpy.array(start, subok=True, copy=False).dtype,
                numpy.array(step, subok=True, copy=False).dtype,
            )
        start = Quantity(start, dtype=dtype, copy=False)
        step = Quantity(step, dtype=dtype, copy=False).to(start.unit)
        stop = start + step * num
        return cls(
            numpy.arange(start.value, stop.value, step.value, dtype=dtype),
            unit=start.unit,
            copy=False,
        )

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
        item = super().__getitem__(key)
        if item.isscalar:
            return item.view(Quantity)
        return item
