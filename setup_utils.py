# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Packaging utilities for the GWpy package
"""

import re
import subprocess
from datetime import date
from itertools import groupby

COPYRIGHT_REGEX = re.compile(
    r"Copyright[\S \t]+\((?P<years>\d\d\d\d([, \d-]+)?)\)",
)
CURRENT_YEAR = date.today().year


def _parse_years(years):
    """Parse string of ints include ranges into a `list` of `int`

    Source: https://stackoverflow.com/a/6405228/1307974
    """
    result = []
    for part in years.split(','):
        if '-' in part:
            a, b = part.split('-')
            a, b = int(a), int(b)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result


def _format_years(years):
    """Format a list of ints into a string including ranges

    Source: https://stackoverflow.com/a/9471386/1307974
    """
    def sub(x):
        return x[1] - x[0]

    ranges = []
    for k, iterable in groupby(enumerate(sorted(years)), sub):
        rng = list(iterable)
        if len(rng) == 1:
            s = str(rng[0][1])
        else:
            s = "{}-{}".format(rng[0][1], rng[-1][1])
        ranges.append(s)
    return ", ".join(ranges)


def update_copyright(path, year=CURRENT_YEAR):
    """Update a file's copyright statement to include the given year
    """
    with open(path, "r") as fobj:
        text = fobj.read().rstrip()
    match = COPYRIGHT_REGEX.search(text)
    x = match.start("years")
    y = match.end("years")
    if text[y-1] == " ":  # don't strip trailing whitespace
        y -= 1
    yearstr = match.group("years")
    years = set(_parse_years(yearstr)) | {year}
    with open(path, "w") as fobj:
        print(text[:x] + _format_years(years) + text[y:], file=fobj)


def update_all_copyright(year=CURRENT_YEAR):
    files = subprocess.check_output([
        "git", "grep", "-l", "-E", r"(\#|\*) Copyright",
    ]).strip().splitlines()
    ignore = {
        "gwpy/utils/sphinx/epydoc.py",
        "docs/_static/js/copybutton.js",
    }
    for path in files:
        if path.decode() in ignore:
            continue
        try:
            update_copyright(path, year)
        except AttributeError:
            raise RuntimeError(
                "failed to update copyright for {!r}".format(path),
            )
