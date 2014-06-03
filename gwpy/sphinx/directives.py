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

"""Fancy literalinclude to prepend lines with '>>>'
"""

from docutils.parsers.rst import directives
from sphinx.directives.code import LiteralInclude

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class GWpyLiteralInclude(LiteralInclude):
    """Fancy LiteralInclude to add doctest prefix '>>>' to all lines
    """
    def run(self):
        out = super(GWpyLiteralInclude, self).run()
        for block in out:
            if block.get('source', '').endswith('.py'): 
                for i, child in enumerate(block.children):
                    content = []
                    for l in child.astext().splitlines():
                        if l.startswith('   '):
                            content.append('    %s' % l)
                        else:
                            content.append('>>> %s' % l)
                    block.children[i] = type(child)('\n'.join(content),
                                                    rawsource=child.rawsource)
                    block['language'] = 'python'
                block.rawsource = block.astext()
        return out

directives.register_directive('literalinclude', GWpyLiteralInclude)


def setup(*args, **kwargs):
    pass
