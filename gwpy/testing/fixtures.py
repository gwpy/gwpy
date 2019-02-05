# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018)
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

"""Custom pytest fixtures for GWpy

This module is imported in gwpy.conftest such that all fixtures declared
here are available to test functions/methods by default.

Developer note: **none of the fixtures here should declare autouse=True**.
"""

import pytest

from matplotlib import rc_context

from ..plot.tex import HAS_TEX
from .utils import TemporaryFilename


# -- I/O ---------------------------------------------------------------------

@pytest.fixture
def tmpfile():
    """Return a temporary filename using `tempfile.mktemp`.

    The fixture **does not create the named file**, but will delete it
    when the test exists if it was created in the mean time.
    """
    with TemporaryFilename() as tmp:
        yield tmp


# -- plotting -----------------------------------------------------------------

SKIP_TEX = pytest.mark.skipif(not HAS_TEX, reason='TeX is not available')


@pytest.fixture(scope='function', params=[
    pytest.param(False, id='no-tex'),
    pytest.param(True, id='usetex', marks=SKIP_TEX)
])
def usetex(request):
    """Repeat a test with matplotlib's `text.usetex` param False and True.

    If TeX is not available on the test machine (determined by
    `gwpy.plot.tex.has_tex()`), the usetex=True tests will be skipped.
    """
    use_ = request.param
    with rc_context(rc={'text.usetex': use_}):
        yield use_


# -- other -------------------------------------------------------------------
