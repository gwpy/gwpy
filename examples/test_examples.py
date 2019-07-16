# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019)
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
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

from matplotlib import use

from gwpy.io.nds2 import NDSWarning

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

if sys.version_info < (3, 6):  # python < 3.6
    pytest.skip('example tests will only run on python >= 3.6',
                allow_module_level=True)
pytest.importorskip("matplotlib", minversion="2.2.0")
pytest.importorskip("astropy", minversion="2.0.0")

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
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


@pytest.fixture(autouse=True)
def example_context():
    from matplotlib import pyplot
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=NDSWarning)
            warnings.filterwarnings('ignore', message=".*non-GUI backend.*")
            yield
    finally:
        # close all open figures regardless of test status
        pyplot.close('all')


@pytest.mark.parametrize('script', EXAMPLES)
def test_example(script):
    with cwd(script.parent):
        with script.open('r') as example:
            raw = example.read()
        if not isinstance(raw, bytes):  # python < 3
            raw = raw.encode('utf-8')
        code = compile(raw, str(script), "exec")
        try:
            exec(code, globals())
        except NDSWarning as exc:  # if we can't authenticate, dont worry
            for msg in NDS2_AUTH_FAILURES:
                if msg.match(str(exc)):
                    pytest.skip(str(exc))
            raise
        except ImportError as exc:  # needs an optional dependency
            if "gwpy" in str(exc):
                raise
            pytest.skip(str(exc))
