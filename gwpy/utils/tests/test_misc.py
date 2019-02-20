# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Tests for :mod:`gwpy.utils.misc`
"""

import sys

import pytest

from .. import misc as utils_misc


def test_gprint(capsys):
    """Test for :func:`gwpy.utils.misc.gprint`
    """
    utils_misc.gprint('test')
    assert capsys.readouterr().out == 'test\n'
    utils_misc.gprint('test', end=' ')
    assert capsys.readouterr().out == 'test '
    utils_misc.gprint('test', end='x', file=sys.stderr)
    cap = capsys.readouterr()
    assert not cap.out
    assert cap.err == 'testx'


def test_null_context():
    """Test for :func:`gwpy.utils.misc.null_context`
    """
    ctx = utils_misc.null_context()
    with ctx:
        print('this should work')


@pytest.mark.parametrize('func, value, out', [
    (str, None, None),
    (str, 1, '1'),
])
def test_if_not_none(func, value, out):
    """Test for :func:`gwpy.utils.misc.if_not_none`
    """
    assert utils_misc.if_not_none(func, value) == out
