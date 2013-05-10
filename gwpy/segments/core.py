# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines the segment and segmentlist objects, as well as the
infinity object used to define semi-infinite and infinite segments.
"""

from bisect import (bisect_left, bisect_right)
from copy import (copy as shallowcopy)

from astropy.io import registry as io_registry

from .. import version

__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = version.version

from .__segments import Segment, SegmentList

__all__ = ["Segment", "SegmentList", "SegmentListDict"]

class SegmentListDict(dict):
    """
    A dictionary associating a unique label and numeric offset with
    each of a set of segmentlist objects.

    This class implements a standard mapping interface, with additional
    features added to assist with the manipulation of a collection of
    segmentlist objects.  In particular, methods for taking unions and
    intersections of the lists in the dictionary are available, as well
    as the ability to record and apply numeric offsets to the
    boundaries of the segments in each list.

    The numeric offsets are stored in the "offsets" attribute, which
    itself is a dictionary, associating a number with each key in the
    main dictionary.  Assigning to one of the entries of the offsets
    attribute has the effect of shifting the corresponding segmentlist
    from its original position (not its current position) by the given
    amount.

    Example:

    >>> x = SegmentListDict()
    >>> x["H1"] = segmentlist([segment(0, 10)])
    >>> print x
    {'H1': [segment(0, 10)]}
    >>> x.offsets["H1"] = 6
    >>> print x
    {'H1': [segment(6.0, 16.0)]}
    >>> x.offsets.clear()
    >>> print x
    {'H1': [segment(0.0, 10.0)]}
    >>> x["H2"] = segmentlist([segment(5, 15)])
    >>> x.intersection(["H1", "H2"])
    [segment(5, 10.0)]
    >>> x.offsets["H1"] = 6
    >>> x.intersection(["H1", "H2"])
    [segment(6.0, 15)]
    >>> c = x.extract_common(["H1", "H2"])
    >>> c.offsets.clear()
    >>> c
    {'H2': [segment(6.0, 15)], 'H1': [segment(0.0, 9.0)]}
    """
    def __new__(cls, *args):
        self = dict.__new__(cls, *args)
        self.offsets = _offsets(self)
        return self

    def __init__(self, *args):
        dict.__init__(self, *args)
        dict.clear(self.offsets)
        for key in self:
            dict.__setitem__(self.offsets, key, 0.0)
        if args and isinstance(args[0], self.__class__):
            dict.update(self.offsets, args[0].offsets)

    def copy(self, keys = None):
        """
        Return a copy of the SegmentListDict object.  The return
        value is a new object with a new offsets attribute, with
        references to the original keys, and shallow copies of the
        segment lists.  Modifications made to the offset dictionary
        or segmentlists in the object returned by this method will
        not affect the original, but without using much memory
        until such modifications are made.  If the optional keys
        argument is not None, then should be an iterable of keys
        and only those segmentlists will be copied (KeyError is
        raised if any of those keys are not in the
        SegmentListDict).

        More details.  There are two "built-in" ways to create a
        copy of a segmentlist object.  The first is to initialize a
        new object from an existing one with

        >>> old = SegmentListDict()
        >>> new = SegmentListDict(old)

        This creates a copy of the dictionary, but not of its
        contents.  That is, this creates new with references to the
        segmentlists in old, therefore changes to the segmentlists
        in either new or old are reflected in both.  The second
        method is

        >>> new = old.copy()

        This creates a copy of the dictionary and of the
        segmentlists, but with references to the segment objects in
        the original segmentlists.  Since segments are immutable,
        this effectively creates a completely independent working
        copy but without the memory cost of a full duplication of
        the data.
        """
        if keys is None:
            keys = self
        new = self.__class__()
        for key in keys:
            new[key] = shallowcopy(self[key])
            dict.__setitem__(new.offsets, key, self.offsets[key])
        return new

    def __setitem__(self, key, value):
        """
        Set the segmentlist associated with a key.  If key is not
        already in the dictionary, the corresponding offset is
        initialized to 0.0, otherwise it is left unchanged.
        """
        dict.__setitem__(self, key, value)
        if key not in self.offsets:
            dict.__setitem__(self.offsets, key, 0.0)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        dict.__delitem__(self.offsets, key)

    # supplementary accessors

    def map(self, func):
        """
        Return a dictionary of the results of func applied to each
        of the segmentlist objects in self.

        Example:

        >>> x = SegmentListDict()
        >>> x["H1"] = segmentlist([segment(0, 10)])
        >>> x["H2"] = segmentlist([segment(5, 15)])
        >>> x.map(lambda l: 12 in l)
        {'H2': True, 'H1': False}
        """
        return dict((key, func(value)) for key, value in self.iteritems())

    def __abs__(self):
        """
        Return a dictionary of the results of running .abs() on
        each of the segmentlists.
        """
        return self.map(abs)

    def extent(self):
        """
        Return a dictionary of the results of running .extent() on
        each of the segmentlists.
        """
        return self.map(segmentlist.extent)

    def extent_all(self):
        """
        Return the result of running .extent() on the union of all
        lists in the dictionary.
        """
        segs = tuple(seglist.extent() for seglist in self.values() if seglist)
        if not segs:
            raise ValueError("empty list")
        return segment(min(seg[0] for seg in segs), max(seg[1] for seg in segs))

    def find(self, item):
        """
        Return a dictionary of the results of running .find() on
        each of the segmentlists.

        Example:

        >>> x = SegmentListDict()
        >>> x["H1"] = segmentlist([segment(0, 10)])
        >>> x["H2"] = segmentlist([segment(5, 15)])
        >>> x.find(7)
        {'H2': 0, 'H1': 0}

        NOTE:  all segmentlists must contain the item or KeyError
        is raised.
        """
        return self.map(lambda x: x.find(item))

    def keys_at(self, x):
        """
        Return a list of the keys for the segment lists that
        contain x.

        Example:

        >>> x = SegmentListDict()
        >>> x["H1"] = segmentlist([segment(0, 10)])
        >>> x["H2"] = segmentlist([segment(5, 15)])
        >>> x.keys_at(12)
        ['H2']
        """
        return [key for key, segs in self.items() if x in segs]

    # list-by-list arithmetic

    def __iand__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] &= value
            else:
                self[key] = segmentlist()
        return self

    def __and__(self, other):
        if sum(len(s) for s in self.values()) <= sum(len(s) for s in other.values()):
            return self.copy().__iand__(other)
        return other.copy().__iand__(self)

    def __ior__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] |= value
            else:
                self[key] = shallowcopy(value)
        return self

    def __or__(self, other):
        if sum(len(s) for s in self.values()) >= sum(len(s) for s in other.values()):
            return self.copy().__ior__(other)
        return other.copy().__ior__(self)

    __iadd__ = __ior__
    __add__ = __or__

    def __isub__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] -= value
        return self

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __ixor__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] ^= value
            else:
                self[key] = shallowcopy(value)
        return self

    def __xor__(self, other):
        if sum(len(s) for s in self.values()) <= sum(len(s) for s in other.values()):
            return self.copy().__ixor__(other)
        return other.copy().__ixor__(self)

    def __invert__(self):
        new = self.copy()
        for key, value in new.items():
            dict.__setitem__(new, key, ~value)
        return new

    # other list-by-list operations

    def intersects_segment(self, seg):
        """
        Returns True if any segmentlist in self intersects the
        segment, otherwise returns False.
        """
        return any(value.intersects_segment(seg) for value in self.itervalues())

    def intersects(self, other):
        """
        Returns True if there exists a segmentlist in self that
        intersects the corresponding segmentlist in other;  returns
        False otherwise.

        See also:

        .intersects_all(), .all_intersects(), .all_intersects_all()
        """
        return any(key in self and self[key].intersects(value) for key, value in other.iteritems())

    def intersects_all(self, other):
        """
        Returns True if each segmentlist in other intersects the
        corresponding segmentlist in self;  returns False
        if this is not the case, or if other is empty.

        See also:

        .intersects(), .all_intersects(), .all_intersects_all()
        """
        return all(key in self and self[key].intersects(value) for key, value in other.iteritems()) and bool(other)

    def all_intersects(self, other):
        """
        Returns True if each segmentlist in self intersects the
        corresponding segmentlist in other;  returns False
        if this is not the case or if self is empty.

        See also:

        .intersects, .intersects_all(), .all_intersects_all()
        """
        return all(key in other and other[key].intersects(value) for key, value in self.iteritems()) and bool(self)

    def all_intersects_all(self, other):
        """
        Returns True if self and other have the same keys, and each
        segmentlist intersects the corresponding segmentlist in the
        other;  returns False if this is not the case or if either
        dictionary is empty.

        See also:

        .intersects(), .all_intersects(), .intersects_all()
        """
        return set(self) == set(other) and all(other[key].intersects(value) for key, value in self.iteritems()) and bool(self)

    def extend(self, other):
        """
        Appends the segmentlists from other to the corresponding
        segmentlists in self, adding new segmentslists to self as
        needed.
        """
        for key, value in other.iteritems():
            if key not in self:
                self[key] = shallowcopy(value)
            else:
                self[key].extend(value)

    def coalesce(self):
        """
        Run .coalesce() on all segmentlists.
        """
        for value in self.itervalues():
            value.coalesce()
        return self

    def contract(self, x):
        """
        Run .contract(x) on all segmentlists.
        """
        for value in self.itervalues():
            value.contract(x)
        return self

    def protract(self, x):
        """
        Run .protract(x) on all segmentlists.
        """
        for value in self.itervalues():
            value.protract(x)
        return self

    def extract_common(self, keys):
        """
        Return a new SegmentListDict containing only those
        segmentlists associated with the keys in keys, with each
        set to their mutual intersection.  The offsets are
        preserved.
        """
        keys = set(keys)
        new = self.__class__()
        intersection = self.intersection(keys)
        for key in keys:
            dict.__setitem__(new, key, shallowcopy(intersection))
            dict.__setitem__(new.offsets, key, self.offsets[key])
        return new

    # multi-list operations

    def is_coincident(self, other, keys = None):
        """
        Return True if any segment in any list in self intersects
        any segment in any list in other.  If the optional keys
        argument is not None, then it should be an iterable of keys
        and only segment lists for those keys will be considered in
        the test (instead of raising KeyError, keys not present in
        both segment list dictionaries will be ignored).  If keys
        is None (the default) then all segment lists are
        considered.

        This method is equivalent to the intersects() method, but
        without requiring the keys of the intersecting segment
        lists to match.
        """
        if keys is not None:
            keys = set(keys)
            self = tuple(self[key] for key in set(self) & keys)
            other = tuple(other[key] for key in set(other) & keys)
        else:
            self = tuple(self.values())
            other = tuple(other.values())
        # make sure inner loop is smallest
        if len(self) < len(other):
            self, other = other, self
        return any(a.intersects(b) for a in self for b in other)

    def intersection(self, keys):
        """
        Return the intersection of the segmentlists associated with
        the keys in keys.
        """
        keys = set(keys)
        if not keys:
            return segmentlist()
        seglist = shallowcopy(self[keys.pop()])
        for key in keys:
            seglist &= self[key]
        return seglist

    def union(self, keys):
        """
        Return the union of the segmentlists associated with the
        keys in keys.
        """
        keys = set(keys)
        if not keys:
            return segmentlist()
        seglist = shallowcopy(self[keys.pop()])
        for key in keys:
            seglist |= self[key]
        return seglist


# setup pickle
import copy_reg

copy_reg.pickle(Segment, lambda x: (segment, tuple(x)))
copy_reg.pickle(SegmentList, lambda x: (segmentlist, (), None, iter(x)))
