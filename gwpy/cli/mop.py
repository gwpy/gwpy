# Copyright (C) Duncan Macleod (2013)
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

"""Parsing for mathematical operations given on the command-line
"""

from ..version import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version

import operator
from StringIO import StringIO
import token
from tokenize import generate_tokens

OPERATORS = {'<': operator.lt,
             '<=': operator.le,
             '=': operator.eq,
             '>=': operator.ge,
             '>': operator.gt,
             '==': operator.is_,
             '!=': operator.is_not}
INV_OPERATORS = {'<': operator.ge,
                 '<=': operator.gt,
                 '=': operator.ne,
                 '>=': operator.lt,
                 '>': operator.le,
                 '==': operator.is_not,
                 '!=': operator.is_}


def parse_math_operation(opstring):
    """Parse a mathematical operation string into a list of variable,
    one or two operators, and matching thresholds thresholds
    """
    splitcon = [tok[0:2] for tok in
                generate_tokens(StringIO(opstring).readline) if tok[1]]
    # parse variable name
    variables = [a[1] for a in splitcon if a[0] in [token.STRING, token.NAME]]
    if len(variables) != 1:
        raise ValueError("Found multiple variables in operation string, "
                         "cannot parse")
    # parse limits
    limits = [a[1] for a in splitcon if token.tok_name[a[0]] == 'NUMBER']
    # parse operators
    operators = [a[1] for a in splitcon if token.tok_name[a[0]] == 'OP']

    # test correct syntax
    if len(limits) != len(operators):
        raise ValueError("Extracted %s limits and %s conditions, please "
                         "reenter" % (len(limits), len(operators)))

    # re-direct operator to put the variable on the left
    for i in range(len(limits)):
        # test direction
        if opstring.find(limits[i]) < opstring.find(operators[i]):
            operators[i] = INV_OPERATORS[operators[i]]
        else:
            operators[i] = OPERATORS[operators[i]]

    # try to format the limits
    for i,l in enumerate(limits):
        try:
            limits[i] = float(l)
        except ValueError:
            pass

    return variables[0], zip(operators, limits)
