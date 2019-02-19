# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

from __future__ import print_function

import sys
from contextlib import contextmanager

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def gprint(*values, **kwargs):  # pylint: disable=missing-docstring
    kwargs.setdefault('file', sys.stdout)
    file_ = kwargs['file']
    print(*values, **kwargs)
    file_.flush()


gprint.__doc__ = print.__doc__


@contextmanager
def null_context():
    """Null context manager
    """
    yield


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
