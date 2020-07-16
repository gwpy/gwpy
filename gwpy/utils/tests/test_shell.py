# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Unit test for utils module
"""

import platform
import subprocess
import sys
from distutils.spawn import find_executable

import pytest

from .. import shell

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_shell_call():
    out, err = shell.call([sys.executable, "--version"])
    assert out.rstrip() == 'Python {}'.format(platform.python_version())
    assert err.rstrip() == ''


def test_shell_call_shell():
    out, err = shell.call("echo This works")
    assert out.rstrip() == "This works"
    assert err.rstrip() == ""


def test_shell_call_errors():
    with pytest.raises(OSError):
        shell.call(['this-command-doesnt-exist'])
    with pytest.raises(subprocess.CalledProcessError):
        shell.call('this-command-doesnt-exist')
    with pytest.raises(subprocess.CalledProcessError):
        shell.call('python --blah')
    with pytest.warns(UserWarning):
        shell.call('python --blah', on_error='warn')


def test_which():
    assert shell.which('python') == find_executable('python')
    with pytest.raises(ValueError):
        shell.which('gwpy-no-executable')
