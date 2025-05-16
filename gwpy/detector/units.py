# Copyright (c) 2017-2025 Cardiff University
#               2014-2017 Louisiana State University
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

"""Custom units and formatting."""

import contextlib
import re

from astropy import (
    __version__ as astropy_version,
    units,
)
from astropy.units import imperial as units_imperial
from astropy.units.format.generic import Generic
from packaging.version import Version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

#: Is the current version of Astropy 7.1 or later?
ASTROPY_71 = Version(astropy_version) >= Version("7.1.0a0")

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

    name = "gwpy"
    re_closest_unit = re.compile(r"Did you mean (.*)\?\Z")
    re_closest_unit_delim = re.compile("(, | or )")

    @classmethod
    def _validate_unit(cls, unit, detailed_exception=True):
        """Validate a unit string."""
        try:
            return super()._validate_unit(unit, detailed_exception)
        except ValueError as exc:
            singular = unit[:-1] if unit.endswith("s") else ""

            # parse alternative units from the error message
            # split 'A, B, or C' -> ['A', 'B', 'C']
            if match := cls.re_closest_unit.search(str(exc)):
                alts = set(cls.re_closest_unit_delim.split(match.groups()[0])[::2])
            else:
                alts = set()

            candidates = list(filter(None, (
                # match uppercase to titled (e.g. MPC -> Mpc)
                unit.title(),
                # match titled unit to lower-case (e.g. Amp -> amp)
                unit.lower(),
                # match plural to singular (e.g. meters -> meter)
                singular,
                singular.lower() if singular else None,
            )))

            for candidate in candidates:
                if candidate in alts:
                    return super()._validate_unit(candidate, detailed_exception)

            raise

    if not ASTROPY_71:
        @classmethod
        def _get_unit(cls, t):
            # match as normal
            try:
                return cls._parse_unit(t.value)
            except ValueError as exc:
                name = t.value
                singular = name[:-1] if name.endswith("s") else ""

                # parse alternative units from the error message
                # split 'A, B, or C' -> ['A', 'B', 'C']
                if match := cls.re_closest_unit.search(str(exc)):
                    alts = set(cls.re_closest_unit_delim.split(match.groups()[0])[::2])
                else:
                    alts = set()

                candidates = list(filter(None, (
                    # match uppercase to titled (e.g. MPC -> Mpc)
                    name.title(),
                    # match titled unit to lower-case (e.g. Amp -> amp)
                    name.lower(),
                    # match plural to singular (e.g. meters -> meter)
                    singular,
                    singular.lower() if singular else None,
                )))

                for candidate in candidates:
                    if candidate in alts:
                        return cls._parse_unit(candidate)

                raise


def parse_unit(
    name,
    parse_strict="warn",
    format="gwpy",
):
    """Attempt to intelligently parse a `str` as a `~astropy.units.Unit`.

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

    # have we already handled this new unit?
    with contextlib.suppress(KeyError):
        return UNRECOGNIZED_UNITS[name]

    # no, either a valid unit, or something new
    try:
        return units.Unit(name, parse_strict="raise")
    except ValueError as exc:
        if (
            # the format was selected by the user
            format in {None, "generic"}
            # or we were asked to be strict about things
            or parse_strict == "raise"
            # or this isn't the error we're looking for
            or "did not parse as unit" not in str(exc)
        ):
            raise
        # try again using our own lenient parser
        new = units.Unit(name, parse_strict=parse_strict, format=format)
        UNRECOGNIZED_UNITS[name] = new
        return new


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
    # GW observatories like to record 'time' as the unit
    (units.Unit('second'), (
        'time',
        'time (s)',
        'time [s]',
        'Time [sec]',
        'Time (sec)',
        'Seconds',  # GWOSC
    )),
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
units.def_unit('coherence', represents=units.dimensionless_unscaled, namespace=_ns)

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
