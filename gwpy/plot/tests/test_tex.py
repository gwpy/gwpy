# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2019)
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

"""Tests for `gwpy.plot.tex`
"""

import pytest

from .. import tex as plot_tex
from ...testing.compat import mock


def which_patcher(error=False):
    def which_(arg):
        if error:
            raise ValueError
        return arg
    return which_


@pytest.mark.parametrize('error', (False, True))
def test_has_tex(error):
    with mock.patch('gwpy.plot.tex.which', side_effect=which_patcher(error)):
        assert plot_tex.has_tex() is not error


@pytest.mark.parametrize('in_, out', [
    (1, '1'),
    (100, r'10^{2}'),
    (-500, r'-5\!\!\times\!\!10^{2}'),
    (0.00001, r'10^{-5}'),
])
def test_float_to_latex(in_, out):
    assert plot_tex.float_to_latex(in_) == out


@pytest.mark.parametrize('in_, out', [
    (None, ''),
    ('normal text', 'normal text'),
    (r'$1 + 2 = 3$', r'$1 + 2 = 3$'),
    ('H1:ABC-DEF_GHI', r'H1:ABC-DEF\_GHI'),
    (r'H1:ABC-DEF\_GHI', r'H1:ABC-DEF\_GHI'),
])
def test_label_to_latex(in_, out):
    assert plot_tex.label_to_latex(in_) == out
