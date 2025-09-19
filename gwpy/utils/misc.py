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

"""Miscellaneous utilties for GWpy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )
    from typing import TypeVar

    T = TypeVar("T")
    R = TypeVar("R")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def if_not_none(
    func: Callable[[T], R],
    value: T,
) -> R | None:
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
        return None
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
    elif which is None:
        selector = round
    else:
        msg = "'which' argument must be one of 'lower', 'upper', or None"
        raise ValueError(msg)
    return type(base)(base ** selector(math.log(x, base)))


def property_alias(
    prop: property,
    doc: str | None = None,
) -> property:
    """Create a property alias for another property.

    This is useful when you want to expose a property from a contained
    object as a property of the containing object, but want to be able
    to set a different docstring for the alias.

    Parameters
    ----------
    prop : `property`
        The property to alias.

    doc : `str`, optional
        The docstring for the new property.  If not given, the docstring
        of the original property will be used.

    Returns
    -------
    alias : `property`
        The new property which acts as an alias for `prop`.

    Examples
    --------
    >>> from gwpy.utils.misc import property_alias
    >>> class A:
    ...     @property
    ...     def x(self):
    ...         "The x property"
    ...         return 1
    >>> class B:
    ...     def __init__(self):
    ...         self.a = A()
    ...     a = property_alias(A.x, "Alias for A.x")
    >>> b = B()
    >>> b.a
    1
    >>> B.a.__doc__
    'Alias for A.x'
    """
    return property(
        prop.fget,
        prop.fset,
        prop.fdel,
        doc or prop.__doc__,
    )


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
