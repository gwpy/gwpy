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

"""Miscellaneous utilties for GWpy
"""

import sys
import math
import warnings
from collections import OrderedDict
from contextlib import nullcontext


__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def gprint(*values, **kwargs):  # pylint: disable=missing-docstring
    kwargs.setdefault('file', sys.stdout)
    file_ = kwargs['file']
    print(*values, **kwargs)
    file_.flush()


gprint.__doc__ = print.__doc__


def null_context():
    """Null context manager
    """
    warnings.warn(
        "gwpy.utils.null_context is deprecated and will be removed in "
        "GWpy 3.1.0, please update your code to use "
        "contextlib.nullcontext from the Python standard library (>=3.7)",
        DeprecationWarning,
    )
    return nullcontext()


def if_not_none(func, value):
    """Apply func to value if value is not None

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


def round_to_power(x, base=2, which=None):
    """Round a positive value to the nearest integer power of `base`

    Parameters
    ----------
    x : scalar
        value to round, must be strictly positive

    base : scalar, optional
        base to whose power `x` will be rounded, default: 2

    which : `str` or `NoneType`, optional
        which value to round to, must be one of `'lower'`, `'upper'`, or
        `None` to round to whichever is nearest, default: `None`

    Returns
    -------
    rounded : scalar
        the rounded value

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
    if which == 'lower':
        selector = math.floor
    elif which == 'upper':
        selector = math.ceil
    elif which is not None:
        raise ValueError("'which' argument must be one of 'lower', "
                         "'upper', or None")
    else:
        selector = round
    return type(base)(base ** selector(math.log(x, base)))


def unique(list_):
    """Return a version of the input list with unique elements,
    preserving order

    Examples
    --------
    >>> from gwpy.utils.misc import unique
    >>> unique(['b', 'c', 'a', 'a', 'd', 'e', 'd', 'a'])
    ['b', 'c', 'a', 'd', 'e']
    """
    return list(OrderedDict.fromkeys(list_).keys())
