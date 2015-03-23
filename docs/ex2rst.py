#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Convert GWpy example python files into rst files for sphinx documentation
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org'

import sys
import os
import argparse
import re

METADATA = {
    'author': 'sectionauthor',
    'currentmodule': 'currentmodule',
}


# -----------------------------------------------------------------------------
# parse command line

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('infile', metavar='example.py',
                    help='python file to convert',)
parser.add_argument('outfile', metavar='example.rst', nargs='?',
                    help='rst file to write, default: print to screen')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# parse python file

lines = open(args.infile, 'rb').read().splitlines()
output = []
header = []

indoc = False
incode = False


for i,line in enumerate(lines):
    # skip file header
    if len(output) == 0 and line.startswith('#'):
        continue

    # end on plot display
    if line.startswith(('if __name__ == ', '# Show')):
        break

    # hide lines
    if line.endswith('# hide'):
        continue

    # find block docs
    if line.startswith('"""'):
        indoc = not indoc
    line = line.strip('"')

    # skip empty lines not in a block quote
    if not line and not indoc:
        continue

    # find code
    if incode and line.startswith(('"', '#', '__')):
        incode = False
        output.append('')

    # find plots

    # comments
    if line.startswith('#'):
        output.append(line.strip('# '))
    # metadata
    elif line.startswith('__'):
        key, value = map(lambda x: x.strip(' _="\'').rstrip(' _="\''),
                         line.split('=', 1))
        if key in METADATA:
            header.append('.. %s:: %s\n' % (METADATA[key], value))
    # block quote
    elif indoc:
        output.append(line.strip('"').rstrip('"'))
    # code
    else:
        if not incode:
            output.append('')
        output.append('    >>> %s' % line)
        incode = True

    # end block quote
    if line.endswith('"""') and indoc:
        output.append('')
        indoc = False

    if len(output) == 1:
        output.append('#'*len(output[0]))

output.append('\n.. plot:: %s\n' % args.infile)
output = header + output

if args.outfile:
    with open(args.outfile, 'w') as f:
        f.write('\n'.join(output))
else:
    print('\n'.join(output))

