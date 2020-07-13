# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Environment utilities for GWpy
"""

import os

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TRUE = (
    '1',
    'y',
    'yes',
    'true',
)


def bool_env(key, default=False):
    """Parse an environment variable as a boolean switch

    `True` is returned if the variable value matches one of the following:

    - ``'1'``
    - ``'y'``
    - ``'yes'``
    - ``'true'``

    The match is case-insensitive (so ``'Yes'`` will match as `True`)

    Parameters
    ----------
    key : `str`
        the name of the environment variable to find

    default : `bool`
        the default return value if the key is not found

    Returns
    -------
    True
        if the environment variable matches as 'yes' or similar
    False
        otherwise

    Examples
    --------
    >>> import os
    >>> from gwpy.utils.env import bool_env
    >>> os.environ['GWPY_VALUE'] = 'yes'
    >>> print(bool_env('GWPY_VALUE'))
    True
    >>> os.environ['GWPY_VALUE'] = 'something else'
    >>> print(bool_env('GWPY_VALUE'))
    False
    >>> print(bool_env('GWPY_VALUE2'))
    False
    """
    try:
        return os.environ[key].lower() in TRUE
    except KeyError:
        return default
