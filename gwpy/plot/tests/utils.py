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

"""Utilities for testing `gwpy.plot`."""

from __future__ import annotations

from io import BytesIO

import pytest
from matplotlib import pyplot
from matplotlib.axes import Axes

from .. import Plot


@pytest.mark.usefixtures("usetex")
class _Base:
    """Base class for tests of `Plot` and `Axes`."""

    @staticmethod
    def save(fig):
        """Save a figure to a 'file' in memory."""
        out = BytesIO()
        fig.savefig(out)
        return fig

    @classmethod
    def save_and_close(cls, fig):
        """Save a figure to a 'file' in memory and then close it."""
        cls.save(fig)
        try:
            fig.close()
        except AttributeError:
            pyplot.close(fig)
        return fig


class FigureTestBase(_Base):
    """Base class for tests of `Plot`."""

    FIGURE_CLASS: type[Plot] = Plot

    @pytest.fixture
    @classmethod
    def fig(cls):
        """Yield a new instance of `.FIGURE_CLASS`.

        This fixture checks that the figure can be rendered as a PNG (in memory)
        before the test function finishes.
        """
        fig = pyplot.figure(FigureClass=cls.FIGURE_CLASS)
        yield fig
        cls.save_and_close(fig)


class AxesTestBase(_Base):
    """Base class for tests of `Axes`."""

    AXES_CLASS: type[Axes] = Axes

    @pytest.fixture
    @classmethod
    def ax(cls):
        """Yield a new instance of `.AXES_CLASS`.

        This fixture checks that the figure can be rendered as a PNG (in memory)
        before the test function finishes.
        """
        fig = pyplot.figure(FigureClass=getattr(cls, "FIGURE_CLASS", Plot))
        yield fig.add_subplot(projection=cls.AXES_CLASS.name)
        cls.save_and_close(fig)
