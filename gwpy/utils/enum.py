# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Utilties for enumerations
"""

from enum import Enum

import numpy

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class NumpyTypeEnum(Enum):
    """`~enum.Enum` of numpy types
    """
    @property
    def dtype(self):
        return numpy.dtype(self.name.lower())

    @property
    def type(self):
        return self.dtype.type

    @classmethod
    def find(cls, type_):
        """Returns the enumerated type corresponding to the given python type
        """
        try:
            return cls(type_)
        except ValueError as exc:
            if isinstance(type_, str):
                type_ = type_.lower()
            try:
                return cls[numpy.dtype(type_).name.upper()]
            except (KeyError, TypeError):
                raise exc
