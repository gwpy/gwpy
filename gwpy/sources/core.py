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

"""This core module provides basic Source and SourceList types
designed to be sub-classes for specific sources
"""

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class Source(object):
    """Generic gravitational-wave source object.

    This class is designed to be sub-classed for specific sources
    """
    pass


class SourceList(list):
    """Generic gravitational-wave source list.

    This class is designed to be sub-classed for specific sources
    """
    def __isub__(self, other):
        """Remove elements from self that are in other.
        """
        end = len(self) - 1
        for i, elem in enumerate(self[::-1]):
            if elem in other:
                del self[end - i]
        return self

    def __sub__(self, other):
        """Return a new list containing the entries of self that
        are not in other.
        """
        return self.__class__([elem for elem in self if elem not in other])

    def __ior__(self, other):
        """Append entries from other onto self without
        introducing (new) duplicates.
        """
        self.extend(other - self)
        return self

    def __or__(self, other):
        """Return a new list containing all entries of self and other.
        """
        return self.__class__(self[:]).__ior__(other)

    def __iand__(self, other):
        """Remove elements in self that are not in other.
        """
        end = len(self) - 1
        for i, elem in enumerate(self[::-1]):
            if elem not in other:
                del self[end - i]
        return self

    def __and__(self, other):
        """Return a new list containing the entries of self that
        are also in other.
        """
        return self.__class__([elem for elem in self if elem in other])

    def unique(self):
        """Return a new list which has every element of self, but without
        duplication.

        Preserves order.
        Does not hash, so a bit slow.
        """
        new = self.__class__([])
        for elem in self:
            if elem not in new:
                new.append(elem)
        return new


