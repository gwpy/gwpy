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

"""This module registers a number of custom units used in GW astronomy.
"""

import re
import warnings

from astropy import units
from astropy.units import imperial as units_imperial
from astropy.units.format.generic import Generic

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# container for new units (so that each one only gets created once)
UNRECOGNIZED_UNITS = {}


# -- parser to handle any unit ------------------------------------------------

class GWpyFormat(Generic):
    """Sub-class of the `Generic` unit parser that is more forgiving

    This format tries to work around 'human' errors in unit naming,
    including plurals, and capitalisation, and if nothing else works
    it just defines a new unit matching the given string.

    New units are not registered, so cannot be referred to later, but are
    created so that mathematical operations will work. Conversions to other
    units will explicitly not work.
    """
    name = 'gwpy'
    re_closest_unit = re.compile(r'Did you mean (.*)\?\Z')
    re_closest_unit_delim = re.compile('(, | or )')
    warn = True

    @classmethod
    def _get_unit(cls, t):
        # match as normal
        try:
            return cls._parse_unit(t.value)
        except ValueError as exc:
            name = t.value
            sname = name[:-1] if name.endswith('s') else ''

            # parse alternative units from the error message
            match = cls.re_closest_unit.search(str(exc))
            try:  # split 'A, B, or C' -> ['A', 'B', 'C']
                alts = cls.re_closest_unit_delim.split(match.groups()[0])[::2]
            except AttributeError:
                alts = []
            alts = list(set(alts))

            # match uppercase to titled (e.g. MPC -> Mpc)
            if name.title() in alts:
                alt = name.title()
            # match titled unit to lower-case (e.g. Amp -> amp)
            elif name.lower() in alts:
                alt = name.lower()
            # match plural to singular (e.g. meters -> meter)
            elif sname in alts:
                alt = sname
            elif sname.lower() in alts:
                alt = sname.lower()
            else:
                if cls.warn:
                    warnings.warn(
                        f"{str(exc).rstrip(' ')} Mathematical operations "
                        "using this unit should work, but conversions to "
                        "other units will not.",
                        category=units.UnitsWarning)
                try:  # return previously created unit
                    return UNRECOGNIZED_UNITS[name]
                except KeyError:  # or create new one now
                    u = UNRECOGNIZED_UNITS[name] = units.def_unit(
                        name, doc='Unrecognized unit')
                    return u
            return cls._parse_unit(alt)


# pylint: disable=redefined-builtin
def parse_unit(name, parse_strict='warn', format='gwpy'):
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
    if name is None or isinstance(name, units.UnitBase):
        return name

    try:  # have we already identified this unit as unrecognised?
        return UNRECOGNIZED_UNITS[name]
    except KeyError:  # no, this is new
        # pylint: disable=unexpected-keyword-arg
        try:
            return units.Unit(name, parse_strict='raise')
        except ValueError as exc:
            if (
                parse_strict == 'raise'
                or 'did not parse as unit' not in str(exc)
            ):
                raise
            # try again using out own lenient parser
            GWpyFormat.warn = parse_strict != 'silent'
            return units.Unit(name, parse_strict='silent', format=format)
        finally:
            GWpyFormat.warn = True


# -- custom units -------------------------------------------------------------
# pylint: disable=no-member,invalid-name

# enable imperial units
units.add_enabled_units(units_imperial)

# -- custom units settings
# the following happens in two sets
#     1) alternative names for standard units where SI prefices will not
#        be used
#     2) new units or alternative names for standard units where SI prefices
#        _will_ be used
#
# for developers: when adding a new custom unit, please remember to add it
# to the list of tested units in `test_detector.py`

# 1) alternative names
registry = units.get_current_unit_registry().registry
for unit, aliases in [
        (units.Unit('ct'), ('counts',)),
        (units.Unit('Celsius'), ('Degrees_C', 'DegC')),
        (units.Unit('Fahrenheit'), ('Degrees_F', 'DegF')),
]:
    unit.names.extend(aliases)
    for alias in aliases:
        registry[alias] = unit

# 2) new units
_ns = {}

# LIGO-Lab standard for 'no unit defined'
units.def_unit(['NONE', 'undef'], namespace=_ns,
               doc='No unit has been defined for these data')

# other dimenionless units
units.def_unit('strain', namespace=_ns)
units.def_unit('coherence', namespace=_ns)

# alias for 'second' but with prefices
units.def_unit((['sec'], ['sec']), represents=units.second, prefixes=True,
               namespace=_ns)

# alternative Pressure unit for LIGO UHV
units.def_unit((['torr'], ['torr']), represents=101325/760.*units.pascal,
               prefixes=True, namespace=_ns)

# pounds per square inch gauge
units.def_unit('psig', represents=units_imperial.psi, prefixes=True,
               namespace=_ns, doc='Pound per square inch gauge: pressure')

# cubic feet
units.def_unit('cf', represents=units_imperial.foot**3, namespace=_ns)

# cubic feet per minute
units.def_unit('cfm', represents=_ns['cf']/units.minute, namespace=_ns)

# particles (as in dust)
units.def_unit(['ptcls', 'particles', 'particulates'], prefixes=True,
               namespace=_ns)

# -- register units -----------------------------
units.add_enabled_units(_ns)
