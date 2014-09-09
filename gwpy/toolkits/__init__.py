# -*- coding: utf-8 -*-
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

"""
Toolkits
========

While GWpy provides a set of convenient representation of gravitational-wave
detector data products, it is built in as general a way as possible.

The GWpy Toolkits mechanism allows users to build their own specific-purpose
extensions of GWpy, and attach them to the same namespace.

For example, if someone wanted to build a ``noisebudget`` extension, for
studying how well the LIGO sensitivity curve can be explained using known
noise components, this can be attached as a 'toolkit' by using python's
namespace package concept.

If you wish to setup a toolkit for GWpy via the namespace package mechanism,
please `open a new GWpy issue on GitHub.com
<https://github.com/gwpy/gwpy/issues>`_.
"""

from .. import version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

__import__('pkg_resources').declare_namespace(__name__)
