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

"""Tests for `gwpy.plot.log`
"""

import pytest

from matplotlib import (__version__ as mpl_version, pyplot, rc_context)

from .. import log as plot_log


@pytest.mark.xfail(mpl_version == '2.0.0', reason='bugs in matplotlib-2.0.0')
@pytest.mark.parametrize('in_, out, texout', [
    (0., r'$\mathdefault{0}$', '$0$'),
    (0.1, r'$\mathdefault{0.1}$', '$0.1$'),
    (1e-5, r'$\mathdefault{10^{-5}}$', r'$10^{-5}$'),
    (1e5, r'$\mathdefault{10^{5}}$', r'$10^{5}$'),
])
def test_log_formatter_mathtext(in_, out, texout):
    formatter = plot_log.GWpyLogFormatterMathtext()
    with rc_context(rc={'text.usetex': False}):
        assert formatter(in_) == out
    with rc_context(rc={'text.usetex': True}):
        assert formatter(in_) == texout


@pytest.mark.parametrize('lim, in_, out', [
    ((1, 11), 5, ''),
    ((1, 9), 5, r'$\mathdefault{5}$'),
])
def test_minor_log_formatter_mathtext(lim, in_, out):
    with rc_context(rc={'text.usetex': False}):
        fig = pyplot.figure()
        ax = fig.gca(yscale='log')
        ax.set_ylim(*lim)
        formatter = ax.yaxis.get_minor_formatter()
        assert isinstance(formatter, plot_log.MinorLogFormatterMathtext)
        assert formatter(in_) == out
        pyplot.close(fig)


@pytest.mark.parametrize('lim, in_, out', [
    ((1, 50), 5, ''),
    ((1, 50), 10, r'$\mathdefault{10}$'),
    ((2, 50), 5, r'$\mathdefault{5}$'),
    ((2, 50), 10, r'$\mathdefault{10}$'),
])
def test_combined_log_formatter_mathtext(lim, in_, out):
    with rc_context(rc={'text.usetex': False}):
        fig = pyplot.figure()
        ax = fig.gca(yscale='log')
        ax.set_ylim(*lim)
        ax.yaxis.set_major_formatter(plot_log.CombinedLogFormatterMathtext())
        formatter = ax.yaxis.get_major_formatter()
        assert formatter(in_) == out
        pyplot.close(fig)
