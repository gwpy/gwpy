
"""Convenience Time representations, used mainly in plotting
"""

from astropy.time import TimeISO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

_HMS = ("hms", "%H:%M:%S", '{hour:02d}:{min:02d}:{sec:02d} ')
_HM = ("hm", "%H:%M", '{hour:02d}:{min:02d} ')
TimeISO.subfmts = tuple(TimeISO.subfmts + (_HMS, _HM))
