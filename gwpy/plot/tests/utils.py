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

"""Utilities for testing `gwpy.plot`
"""

from io import BytesIO

import pytest

from matplotlib import pyplot

from .. import Plot


@pytest.mark.usefixtures("usetex")
class _Base(object):
    @staticmethod
    def save(fig, format='png'):
        out = BytesIO()
        fig.savefig(out, format=format)
        return fig

    @classmethod
    def save_and_close(cls, fig, format='png'):
        cls.save(fig, format=format)
        try:
            fig.close()
        except AttributeError:
            pyplot.close(fig)
        return fig


class FigureTestBase(_Base):
    FIGURE_CLASS = Plot

    @pytest.fixture
    def fig(self):
        """Yield a new figure of type ``FIGURE_CLASS`` and check that
        it saves as png after the test function finishes
        """
        fig = pyplot.figure(FigureClass=self.FIGURE_CLASS)
        yield fig
        self.save_and_close(fig)


class AxesTestBase(_Base):
    AXES_CLASS = Plot

    @pytest.fixture
    def ax(self):
        fig = pyplot.figure(FigureClass=getattr(self, 'FIGURE_CLASS', Plot))
        yield fig.add_subplot(projection=self.AXES_CLASS.name)
        self.save_and_close(fig)
