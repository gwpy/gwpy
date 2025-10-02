# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Quantity array for indexing a Series."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy
from astropy.units import Quantity

from .array import COPY_IF_NEEDED

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import (
        ArrayLike,
        DTypeLike,
    )


class Index(Quantity):
    """1-D `~astropy.units.Quantity` array for indexing a `Series`.

    See Also
    --------
    astropy.units.Quantity
        For parameters supported when creating an `Index` array.
    """

    @classmethod
    def define(
        cls,
        start: float,
        step: float,
        num: int,
        dtype: DTypeLike = None,
    ) -> Self:
        """Define a new `Index` using `numpy.arange`.

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
                numpy.array(start, subok=True, copy=COPY_IF_NEEDED).dtype,
                numpy.array(step, subok=True, copy=COPY_IF_NEEDED).dtype,
            )
        startq = Quantity(start, dtype=dtype, copy=COPY_IF_NEEDED)
        stepq = Quantity(step, dtype=dtype, copy=COPY_IF_NEEDED).to(startq.unit)
        stopq = startq + stepq * num
        return cls(
            numpy.arange(
                startq.value,
                stopq.value,
                stepq.value,
                dtype=dtype,
            )[:num],
            unit=startq.unit,
            copy=COPY_IF_NEEDED,
        )

    @property
    def regular(self) -> bool:
        """`True` if this index is linearly increasing."""
        try:
            return self.info.meta["regular"]
        except (TypeError, KeyError):
            if self.info.meta is None:
                self.info.meta = {}
            self.info.meta["regular"] = self.is_regular()
            return self.info.meta["regular"]

    def is_regular(self) -> bool:
        """Determine whether this `Index` contains linearly increasing samples.

        This also works for linear decrease.
        """
        if self.size <= 1:
            return False
        return bool(numpy.isclose(numpy.diff(self.value, n=2), 0).all())

    def __getitem__(
        self,
        key: slice | int | bool | ArrayLike,
    ) -> Quantity:
        """Get an item or a slice of this `Index`."""
        item = super().__getitem__(key)
        if item.isscalar:
            return item.view(Quantity)
        return item

    def __setitem__(
        self,
        key: slice | int | bool | ArrayLike,
        value: float | ArrayLike,
    ) -> None:
        """Set an element or slice of this `Index`."""
        super().__setitem__(key, value)
        # We have changed the Index, so reset our understanding of regularity.
        with contextlib.suppress(AttributeError, TypeError):
            self.info.meta.pop("regular", None)
