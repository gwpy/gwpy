# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING

from astropy import (
    __version__ as astropy_version,
    units,
)
from astropy.units import imperial as units_imperial
from astropy.units.format.generic import Generic
from packaging.version import Version

if TYPE_CHECKING:
    from typing import Literal

    from astropy.units import UnitBase
    from astropy.units.format.base import LexToken

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

#: Is the current version of Astropy 7.1 or later?
ASTROPY_71 = Version(astropy_version) >= Version("7.1.0a0")

# container for new units (so that each one only gets created once)
UNRECOGNIZED_UNITS: dict[str, units.UnrecognizedUnit] = {}


# -- parser to handle any unit -------

class GWpyFormat(Generic):
    """Sub-class of the `~astropy.units.format.Generic` unit parser that is forgiving.

    This format tries to work around 'human' errors in unit naming,
    including plurals and capitalisation, and if nothing else works
    it just defines a new unit matching the given string.

    New units are not registered, so cannot be referred to later, but are
    created so that mathematical operations will work. Conversions to other
    units will explicitly not work.
    """

    name = "gwpy"
    _re_closest_unit = re.compile(r"Did you mean (.*)\?\Z")
    _re_closest_unit_delim = re.compile("(, | or )")

    @classmethod
    def _get_unit(cls, t: LexToken) -> UnitBase:
        """Get the unit for a lexer token, with lenient matching.

        This method handles the lenient unit parsing for all Astropy versions.
        In Astropy < 7.1, there was no base implementation of this method.
        In Astropy 7.1, this method was removed in favor of `_validate_unit`.
        In Astropy >= 7.2, this method was restored to the base class.
        """
        exc = None
        name = t.value

        # For Astropy >= 7.1, use the base class implementation first
        if ASTROPY_71:
            try:
                return super()._get_unit(t)
            except ValueError as err:
                exc = err
        # For Astropy < 7.1, call _parse_unit directly
        else:
            try:
                return cls._parse_unit(name)
            except ValueError as err:
                exc = err

        # Common fallback logic for all versions
        singular = name[:-1] if name.endswith("s") else ""

        # Parse alternative units from the error message
        # split 'A, B, or C' -> ['A', 'B', 'C']
        if exc and (match := cls._re_closest_unit.search(str(exc))):
            alts = set(cls._re_closest_unit_delim.split(match.groups()[0])[::2])
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
                if ASTROPY_71:
                    # Use _validate_unit for Astropy >= 7.1
                    try:
                        return cls._validate_unit(candidate)
                    except (ValueError, KeyError):
                        continue
                else:
                    # Use _parse_unit for Astropy < 7.1
                    return cls._parse_unit(candidate)

        # Re-raise the original exception if no alternatives worked
        if exc:
            raise exc
        msg = f"Unit '{name}' not recognized"
        raise ValueError(msg)


def parse_unit(
    name: str | UnitBase | None,
    parse_strict: Literal["raise", "warn", "silent"] = "warn",
    format: str = "gwpy",  # noqa: A002
) -> UnitBase:
    """Attempt to intelligently parse a `str` as a `~astropy.units.Unit`.

    Parameters
    ----------
    name : `str`
        Unit name to parse.

    parse_strict : `str`
        One of 'silent', 'warn', or 'raise' depending on how pedantic
        you want the parser to be.

    format : `~astropy.units.format.Base`
        The formatter class to use when parsing the unit string.

    Returns
    -------
    unit : `~astropy.units.UnitBase`
        The unit parsed by `~astropy.units.Unit`.

    Raises
    ------
    ValueError
        If the unit cannot be parsed and `parse_strict='raise'`.
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


# -- custom units --------------------

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
for unit, aliases in {
    units.Unit("ct"): ("counts",),
    units.Unit("Celsius"): ("Degrees_C", "DegC"),
    units.Unit("Fahrenheit"): ("Degrees_F", "DegF"),
    # GW observatories like to record 'time' as the unit
    units.Unit("second"): (
        "time",
        "time (s)",
        "time [s]",
        "Time [sec]",
        "Time (sec)",
        "Seconds",  # GWOSC
    ),
}.items():
    unit.names.extend(aliases)
    for alias in aliases:
        registry[alias] = unit

# 2) new units
_ns: dict[str, UnitBase] = {}

# LIGO-Lab standard for 'no unit defined'
units.def_unit(
    ["NONE", "undef"],
    namespace=_ns,
    doc="No unit has been defined for these data",
)

# other dimenionless units
units.def_unit(
    "strain",
    namespace=_ns,
)
units.def_unit(
    "coherence",
    represents=units.dimensionless_unscaled,
    namespace=_ns,
)

# alias for 'second' but with prefices
units.def_unit(
    (["sec"], ["sec"]),
    represents=units.second,
    prefixes=True,
    namespace=_ns,
)

# alternative Pressure unit for LIGO UHV
units.def_unit(
    (["torr"], ["torr"]),
    represents=101325 / 760. * units.pascal,
    prefixes=True,
    namespace=_ns,
)

# pounds per square inch gauge
units.def_unit(
    "psig",
    represents=units_imperial.psi,
    prefixes=True,
    namespace=_ns,
    doc="Pound per square inch gauge: pressure",
)

# cubic feet
units.def_unit(
    "cf",
    represents=units_imperial.foot ** 3,
    namespace=_ns,
)

# cubic feet per minute
units.def_unit(
    "cfm",
    represents=_ns["cf"] / units.minute,
    namespace=_ns,
)

# particles (as in dust)
units.def_unit(
    ["ptcls", "particles", "particulates"],
    prefixes=True,
    namespace=_ns,
)

# -- register units ------------------

units.add_enabled_units(_ns)
