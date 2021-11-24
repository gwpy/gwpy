# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

from unittest import mock

import pytest

from .. import tex as plot_tex


def _which(arg):
    """Fake which to force pdflatex to being not found
    """
    if arg == "pdflatex":
        return None
    return arg


@mock.patch("gwpy.plot.tex.which", return_value="path")
def test_has_tex_true(_):
    """Test that `gwpy.plot.tex.has_tex` returns `True` when
    all of the necessary executables are found
    """
    assert plot_tex.has_tex()


@mock.patch("gwpy.plot.tex.which", _which)
def test_has_tex_false():
    """Test that `gwpy.plot.tex.has_tex` returns `False` when
    any one of the necessary executables is missing.
    """
    assert not plot_tex.has_tex()


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
