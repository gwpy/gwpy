# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Test suite for all examples
"""

import os
import re
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

from matplotlib import use

from gwpy.io.nds2 import NDSWarning

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

pytest.importorskip("matplotlib", minversion="2.2.0")

use('agg')  # force non-interactive backend

# acceptable authentication failures
NDS2_AUTH_FAILURES = {
    re.compile("failed to establish a connection", re.I),
    re.compile("error authenticating against", re.I),
}

# find all examples
EXAMPLE_BASE = Path(__file__).parent
EXAMPLE_DIRS = [exdir for exdir in EXAMPLE_BASE.iterdir() if exdir.is_dir()]
EXAMPLES = sorted([
    pytest.param(expy, id=str(expy.relative_to(EXAMPLE_BASE))) for
    exdir in EXAMPLE_DIRS for expy in exdir.glob('*.py')],
)


@contextmanager
def cwd(path):
    oldpwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(oldpwd))


@pytest.fixture(autouse=True)
def example_context():
    from matplotlib import pyplot
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=NDSWarning)
            warnings.filterwarnings('ignore', message=".*non-GUI backend.*")
            warnings.filterwarnings("ignore", message="numpy.ufunc size")
            yield
    finally:
        # close all open figures regardless of test status
        pyplot.close('all')


@pytest.mark.parametrize('script', EXAMPLES)
def test_example(script):
    with cwd(script.parent):
        with script.open('r') as example:
            raw = example.read()
        code = compile(raw, str(script), "exec")
        try:
            exec(code, globals())
        except NDSWarning as exc:  # pragma: no-cover
            # if we can't authenticate, dont worry
            for msg in NDS2_AUTH_FAILURES:
                if msg.match(str(exc)):
                    pytest.skip(str(exc))
            raise
        except ImportError as exc:  # pragma: no-cover
            # needs an optional dependency
            if "gwpy" in str(exc):
                raise
            pytest.skip(str(exc))
