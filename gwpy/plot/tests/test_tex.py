# Copyright (c) 2018-2025 Cardiff University
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

"""Tests for `gwpy.plot.tex`."""

from __future__ import annotations

from unittest import mock

import pytest

from .. import tex as plot_tex


def _which(arg: str) -> str | None:
    """Fake which to force pdflatex to being not found.

    Parameters
    ----------
    arg : `str`
        The path to find.

    Returns
    -------
    path : `str` or `None`
        If ``arg`` is ``"pdflatex"`` this function returns `None`,
        otherwise ``arg`` is returned verbatim.
    """
    if arg == "pdflatex":
        return None
    return arg


@mock.patch("gwpy.plot.tex.which", _which)
def test_has_tex_missing_exe():
    """Test `has_tex()` handling of missing executables."""
    plot_tex.has_tex.cache_clear()
    assert not plot_tex.has_tex()


@mock.patch("gwpy.plot.tex._test_usetex", side_effect=RuntimeError)
def test_has_tex_bad_latex(mock_test_usetex):
    """Test `has_tex()` handling of a rendering failure."""
    plot_tex.has_tex.cache_clear()
    assert not plot_tex.has_tex()


@mock.patch("gwpy.plot.tex.which", return_value="path")
@mock.patch("gwpy.plot.tex._test_usetex")
def test_has_tex_true(mock_which_, mock_test_usetex):
    """Test `has_tex()` returns `True` when everything works."""
    plot_tex.has_tex.cache_clear()
    assert plot_tex.has_tex()


@pytest.mark.parametrize(("in_", "out"), [
    pytest.param(
        1,
        "1",
        id="1",
    ),
    pytest.param(
        100,
        r"10^{2}",
        id="power of ten",
    ),
    pytest.param(
        -500,
        r"-5\!\!\times\!\!10^{2}",
        id="scientific notation",
    ),
    pytest.param(
        0.00001,
        r"10^{-5}",
        id="negative power of ten",
    ),
])
def test_float_to_latex(in_, out):
    """Test `float_to_latex()`."""
    assert plot_tex.float_to_latex(in_) == out


@pytest.mark.parametrize(("in_", "out"), [
    pytest.param(
        None,
        "",
        id="None",
    ),
    pytest.param(
        "normal text",
        "normal text",
        id="text",
    ),
    pytest.param(
        r"$1 + 2 = 3$",
        r"$1 + 2 = 3$",
        id="mathtex",
    ),
    pytest.param(
        "H1:ABC-DEF_GHI",
        r"H1:ABC-DEF\_GHI",
        id="underscore",
    ),
    pytest.param(
        r"H1:ABC-DEF\_GHI",
        r"H1:ABC-DEF\_GHI",
        id="escaped",
    ),
])
def test_label_to_latex(in_, out):
    """Test `label_to_latex()`."""
    assert plot_tex.label_to_latex(in_) == out
