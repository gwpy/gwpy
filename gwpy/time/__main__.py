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

"""Primitive command-line interface to `gwpy.time.tconvert`.

Either pass a GPS time to convert to a date string, or a date string
to convert to a GPS time.
"""

import argparse
import datetime
import sys

from dateutil import tz

from .. import __version__
from . import tconvert


def main(args=None):
    """Parse command-line arguments, tconvert inputs, and print
    """
    # define command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-V", "--version", action="version",
                        version=__version__,
                        help="show version number and exit")
    parser.add_argument("-l", "--local", action="store_true", default=False,
                        help="print datetimes in local timezone")
    parser.add_argument("-f", "--format", type=str, action="store",
                        default=r"%Y-%m-%d %H:%M:%S.%f %Z",
                        help="output datetime format (default: %(default)r)")
    parser.add_argument("input", help="GPS or datetime string to convert",
                        nargs="*")

    # parse and convert
    args = parser.parse_args(args)
    input_ = " ".join(args.input)
    output = tconvert(input_)

    # print (now with timezones!)
    if isinstance(output, datetime.datetime):
        output = output.replace(tzinfo=tz.tzutc())
        if args.local:
            output = output.astimezone(tz.tzlocal())
        print(output.strftime(args.format))
    else:
        print(output)


if __name__ == "__main__":  # pragma: no-cover
    sys.exit(main())
