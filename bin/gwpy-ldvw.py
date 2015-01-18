#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Joseph Areeda (2015)
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
#
"""Command line interface to GWpy plotting functions
"""

__author__ = 'joseph areeda'
__email__ = 'joseph.areeda@ligo.org'
__version__ = '0.0.0'

VERBOSE = 1 # 0 = errors only, 1 = Warnings, 2 = INFO, >2 DEBUG >=5 ALL

# each Product calls a function in this file to create the plot
def coher(args):
    if VERBOSE > 1:
        print 'coherence called'
    mod = import_module('%s' % 'Coher')
    class_ = getattr(mod, 'Coher')

    plotObj = class_()
    plotObj.makePlot(plotObj, args)

import sys
if sys.version < '2.6':
    raise ImportError("Python versions older than 2.6 are not supported.")

from importlib import import_module
import argparse

#---needed to generate help messages---
class CliHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('indent_increment', 4)
        super(CliHelpFormatter, self).__init__(*args, **kwargs)
#----

# Products are classes that implement a specific plot
PRODUCTS = [
    'Coher',
    'Spectrum',
    'TimeSeries',
    'Spectrogram',
    'Coherencegram'
]

parser = argparse.ArgumentParser(formatter_class=CliHelpFormatter,
                                 description=__doc__, prog='gwpy_ldvw')
# Setup the argument parser to act as the parent
parentparser = argparse.ArgumentParser(add_help=False)
#,formatter_class=CliHelpFormatter,
#                                       description=__doc__, prog='gwpy_ldvw'

# These arguments apply to all commands
parentparser.add_argument('-v', '--verbose', action='count', default=1,
                          help='increase verbose output')
parentparser.add_argument('-s', '--silent', default=False, help='show only fatal errors')

# subparsers are dependent on which action is chosen
subparsers = parentparser.add_subparsers(
    dest='mode', title='Actions',
    description='Select one of the following actions:')

# dictionary for subparsers
sp = dict()


# -------------------------
# Add the actions and their parameters to the subparsers

# todo ask Duncan how to really do the import, this sucks but it works for testing
path = sys.path
path.insert(1,'/Users/areeda/ligo/gwpy/gwpy/cli')
sys.path = path

# Add the subparsers for each plot product
for product in PRODUCTS:

    mod = import_module('%s' % product)
    class_ = getattr(mod, product)
    prod = class_()

    action = prod.get_action()
    sp[product] = subparsers.add_parser(product, help=prod.__doc__,
                                       parents=[parentparser])
    sp[product].set_defaults(func=product.lower())
    prod.init_cli(sp[product])

# -----------------------------------------------------------------------------
# Run

if __name__ == '__main__':
    import os
    # if we're launched with minimum or no environment variables make some guesses
    if len(os.getenv('HOME', '')) == 0:
        os.environ['HOME'] = '/tmp/'
    # if launched from a terminal with no display
    if len(os.getenv('DISPLAY', '')) == 0:
        import matplotlib
        matplotlib.use('Agg')

    # import all third-party packages in sets (in vague order of dependence)
    import numpy
    from astropy.time import Time
    from matplotlib import rcParams
    from gwpy.timeseries import TimeSeries
    from gwpy.plotter.tex import label_to_latex

    # parse the command line
    args = parentparser.parse_args()
    if args.silent:
        VERBOSE = 0
    else:
        VERBOSE = args.verbose
    if not args.mode:
        raise RuntimeError("Must specify action. Please try again with --help.")
    sys.exit(args.func(args))

