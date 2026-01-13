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

"""Primitive command-line interface to `gwpy.time.tconvert`.

Either pass a GPS time to convert to a date string, or a date string
to convert to a GPS time.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from dateutil import tz

from .. import __version__
from ..tools import _utils
from . import tconvert

if TYPE_CHECKING:
    from argparse import ArgumentParser

EXAMPLES = {
    "Convert GPS time to date string": "gwpy-tconvert 1126259462",
    "Convert date string to GPS time": "gwpy-tconvert 2015-09-14 09:50:45",
    "Find GPS time now": "gwpy-tconvert now",
    "Find GPS time for the start of today": "gwpy-tconvert today",
}


def create_parser() -> ArgumentParser:
    """Parse command-line arguments, tconvert inputs, and print."""
    # define command line arguments
    parser = _utils.ArgumentParser(
        description=__doc__,
        examples=EXAMPLES,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="show version number and exit",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=False,
        help="print datetimes in local timezone",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        action="store",
        default=r"%Y-%m-%d %H:%M:%S.%f %Z",
        help="output datetime format (default: %(default)r)")
    parser.add_argument(
        "input",
        help="GPS or datetime string to convert",
        nargs="*",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run this tool."""
    # parse and convert
    parser = create_parser()
    opts = parser.parse_args(args)
    input_ = " ".join(opts.input)
    output = tconvert(input_)

    # print (now with timezones!)
    if isinstance(output, datetime.datetime):
        output = output.replace(tzinfo=tz.tzutc())
        if opts.local:
            output = output.astimezone(tz.tzlocal())
        print(output.strftime(opts.format))
    else:
        print(output)


if __name__ == "__main__":  # pragma: no-cover
    main()
