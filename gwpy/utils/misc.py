# Copyright (c) 2017-2025 Cardiff University
#               2014-2017 Louisiana State University
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

"""Miscellaneous utilties for GWpy."""

from __future__ import annotations

import math
import typing

if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )
    from typing import (
        Any,
        TypeVar,
    )

    T = TypeVar("T")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def if_not_none(
    func: Callable,
    value: Any,
) -> Any:
    """Apply func to value if value is not None.

    Examples
    --------
    >>> from gwpy.utils.misc import if_not_none
    >>> if_not_none(int, '1')
    1
    >>> if_not_none(int, None)
    None
    """
    if value is None:
        return
    return func(value)


def round_to_power(
    x: float,
    base: float = 2,
    which: str | None = None,
) -> float:
    """Round a positive value to the nearest integer power of `base`.

    Parameters
    ----------
    x : `float`
        Value to round, must be strictly positive.

    base : `float`
        Base to whose power `x` will be rounded.

    which : `str` or `None`
        Which value to round to, must be one of `'lower'`, `'upper'`, or
        `None` to round to whichever is nearest.

    Returns
    -------
    rounded : `float`
        The rounded value.

    Notes
    -----
    The object returned will be of the same type as `base`.

    Examples
    --------
    >>> from gwpy.utils.misc import round_to_power
    >>> round_to_power(2)
    2
    >>> round_to_power(9, base=10)
    10
    >>> round_to_power(5, which='lower')
    4
    """
    selector: Callable
    if which == "lower":
        selector = math.floor
    elif which == "upper":
        selector = math.ceil
    elif which is not None:
        raise ValueError("'which' argument must be one of 'lower', "
                         "'upper', or None")
    else:
        selector = round
    return type(base)(base ** selector(math.log(x, base)))


def unique(
    list_: Iterable[T],
) -> list[T]:
    """Return a version of ``list_`` with unique elements, preserving order.

    Examples
    --------
    >>> from gwpy.utils.misc import unique
    >>> unique(['b', 'c', 'a', 'a', 'd', 'e', 'd', 'a'])
    ['b', 'c', 'a', 'd', 'e']
    """
    return list(dict.fromkeys(list_))
