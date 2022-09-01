#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Joseph Areeda (2015-2020)
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

"""Generate plots of GW observatory data using GWpy
"""

import time

import os
import sys
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    RawTextHelpFormatter,
)

from matplotlib import use

from .. import __version__
from . import PRODUCTS

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'

# if launched from a terminal with no display
# Must be done before modules like pyplot are imported
if len(os.getenv('DISPLAY', '')) == 0:
    use('Agg')

PROG_START = time.time()    # verbose enough times major ops

INTERACTIVE = hasattr(sys, 'ps1')

EPILOG = f"""
Examples:

    $ gwpy-plot timeseries --chan H1:GDS-CALIB_STRAIN --start 1126259457

    $ gwpy-plot spectrum --chan H1:GDS-CALIB_STRAIN L1:GDS-CALIB_STRAIN --chan V1:Hrec_hoft_16384Hz --start 1187008866 --duration 32 --xmin 10 --xmax 4000

    $ gwpy-plot coherencegram --chan H1:GDS-CALIB_STRAIN H1:PEM-CS_ACC_PSL_PERISCOPE_X_DQ --start 1126260017 --duration 600

Written by {__author__}.
Report bugs to https://github.com/gwpy/gwpy/issues/.
"""  # noqa: E501


# -- init command line --------------------------------------------------------

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "Usage: "
        return super()._format_usage(
            usage,
            actions,
            groups,
            prefix,
        )


class _ArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._positionals.title = 'Positional arguments'
        self._optionals.title = 'Options'


def create_parser():
    parser = _ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument('-V', '--version', action='version',
                        version=__version__)

    # set the argument parser to act as the parent
    parentparser = _ArgumentParser(add_help=False)
    parentparser._optionals.title = 'Verbosity options'
    parentparser.add_argument('-v', '--verbose', action='count', default=1,
                              help='increase verbose output')
    parentparser.add_argument('-s', '--silent', action='store_true',
                              help='show only fatal errors')

    # subparsers are dependent on which action is chosen
    subparsers = parser.add_subparsers(
        dest='mode', title='Actions',
        description='Select one of the following actions:')

    # Add the subparsers for each plot product
    for product, product_class in PRODUCTS.items():
        subparser = subparsers.add_parser(
            product, help=product_class.__doc__.strip().split('\n')[0],
            parents=[parentparser],
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        product_class.init_cli(subparser)

    return parser


def parse_command_line(args=None):
    parser = create_parser()
    return parser.parse_args(args=args)


# -- run ----------------------------------------------------------------------

def main(args=None):
    """Run gwpy-plot

    Returns the relevant exit code, that can be passed to :func:`sys.exit`.
    """
    # parse the command line and create a product object
    args = parse_command_line(args=args)
    prod = PRODUCTS[args.mode](args)
    prod.log(2, f'{prod.action} created')

    # log how long it took us to get here
    setup_time = time.time() - PROG_START
    prod.log(2, f'Setup time {setup_time:.1f} sec')

    # -- generate the plot
    prod.run()

    # overload the current namespace for interactive (i)python users
    if INTERACTIVE:
        # import pyplot so that user has access to it
        from matplotlib import pyplot as plt  # noqa: F401
        plot = prod.plot
        # pull raw data and plotted results from product for their use
        timeseries = prod.timeseries  # noqa: F841
        result = prod.result  # noqa: F841
        print('Raw data is in "timeseries", plotted data is in "result"')
        ax = plot.gca()  # noqa: F841

    run_time = time.time() - PROG_START
    prod.log(1, f'Program run time: {run_time:.1f}')
    if prod.got_error:
        return 2     # make sure when running batch they can test for error


if __name__ == "__main__":  # pragma: no-cover
    sys.exit(main())
