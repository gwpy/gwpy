# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

"""Monkeypatch for numpydoc to plot gwpy docstring examples

By default (as of v0.6), numpydoc will only auto-plot code blocks in
_Examples_ sections if they include 'import matplotlib'. Since GWpy examples
typically don't need to import matplotlib, we replace the relevant method
here with one that detects gwpy's typical plot generation and display code,
namely just `plot.show()`.

Hopefully numpydoc will support other key phrases as of v0.7, so this
module could then be deprecated.
"""

from __future__ import absolute_import

try:
    from numpydoc import __version__ as numpydoc_version
except ImportError:  # __version__ only added in 0.7
    numpydoc_version = '0.0'

# this feature will be added to numpydoc in 0.7
if numpydoc_version < '0.7':

    from numpydoc import docscrape_sphinx

    # can't just replace the whole class, SphinxObjDoc complains, so need to
    # swap out just the method we want

    # record the original
    _str_examples_orig = docscrape_sphinx.SphinxDocString._str_examples

    # redefine the class method
    def _str_examples(self):
        examples_str = "\n".join(self['Examples'])
        if self.use_plots and 'plot::' not in examples_str and (
                'plot.show()' in examples_str or
                'plt.show()' in examples_str or
                'fig.show()' in examples_str):
            self['Examples'].insert(0, 'import matplotlib')
            lines = _str_examples_orig(self)
            lines.pop(lines.index('    import matplotlib'))
            return lines
        return _str_examples_orig(self)

     # replace the class method
    docscrape_sphinx.SphinxDocString._str_examples = _str_examples
