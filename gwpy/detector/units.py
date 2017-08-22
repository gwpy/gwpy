# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2015)
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

"""This module registers a number of custom units used in GW astronomy.
"""

import re

from astropy import units
from astropy.units.format.generic import Generic

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- parser to handle plurals -------------------------------------------------


class PluralFormat(Generic):
    """Sub-class of the `Generic` unit parser that handles plurals

    This just enables uses to specify a unit as 'meters' instead of just
    'meter', and have the parse handle things as well as can be expected.
    """
    re_closest_unit = re.compile(r'Did you mean (.*)\?\Z')
    re_closest_unit_delim = re.compile('(, | or )')

    @classmethod
    def _get_unit(cls, t):
        # match as normal
        try:
            return cls._parse_unit(t.value)
        except ValueError as exc:
            # if error message suggests one alternative that is just the
            # singular of the unit given, use it, otherwise re-raise the
            # original exception
            match = cls.re_closest_unit.search(str(exc))
            try:  # split 'A, B, or C' -> ['A', 'B', 'C']
                alts = cls.re_closest_unit_delim.split(match.groups()[0])[::2]
            except AttributeError:
                raise exc
            alts = list(set(map(str.lower, alts)))
            if len(alts) == 1 and '%ss' % alts[0] == t.value.lower():
                try:
                    return cls._parse_unit(alts[0])
                except ValueError:
                    raise exc
            raise exc


# pylint: disable=redefined-builtin
def parse_unit(name, parse_strict='warn', format=PluralFormat):
    """Attempt to intelligently parse a `str` as a `~astropy.units.Unit`

    Parameters
    ----------
    name : `str`
        unit name to parse

    parse_strict : `str`
        one of 'silent', 'warn', or 'raise' depending on how pedantic
        you want the parser to be

    format : `~astropy.units.format.Base`
        the formatter class to use when parsing the unit string

    Returns
    -------
    unit : `~astropy.units.UnitBase`
        the unit parsed by `~astropy.units.Unit`

    Raises
    ------
    ValueError
        if the unit cannot be parsed and `parse_strict='raise'`
    """
    if name is None:
        return None

    # pylint: disable=unexpected-keyword-arg
    return units.Unit(name, parse_strict=parse_strict, format=format)


# -- custom units -------------------------------------------------------------

# enable imperial units
units.add_enabled_units(units.imperial)

# custom GWO units
units.add_enabled_units([
    units.def_unit(['counts'], represents=units.Unit('count')),
    units.def_unit(['undef'], doc='No unit has been defined for these data'),
    units.def_unit(['coherence'], represents=units.dimensionless_unscaled),
    units.def_unit(['strain'], represents=units.dimensionless_unscaled),
    units.def_unit(['Degrees_C'], represents=units.Unit('Celsius')),
    units.def_unit(['Degrees_F'], represents=units.Unit('Fahrenheit')),
])
