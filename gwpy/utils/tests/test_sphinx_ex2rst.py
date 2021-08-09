# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2021)
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

"""Tests for :mod:`gwpy.utils.sphinx.ex2rst`
"""

import pytest

from ..sphinx import ex2rst

EXAMPLE = """
# Header

\"\"\"Example example

Something something something
\"\"\"

__author__ = "Duncan Macleod"
__credits__ = "Someone else"  # ignored
__currentmodule__ = "gwpy.timeseries"

# This is an example:
from gwpy.timeseries import TimeSeries

# and then we do this:
a = TimeSeries(12345)
plot = a.plot()
plot.show()

# and then something else
b = TimeSeries(67890)

# tada!
"""

EXAMPLE_OUTPUT = """
.. _gwpy-example-test_ex2rst0-example:

.. sectionauthor:: Duncan Macleod

.. currentmodule:: gwpy.timeseries

Example example
###############

Something something something

This is an example:

.. plot::
   :context: reset
   :nofigs:
   :include-source:

   from gwpy.timeseries import TimeSeries

and then we do this:

.. plot::
   :context:
   :include-source:

   a = TimeSeries(12345)
   plot = a.plot()
   plot.show()

and then something else

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   b = TimeSeries(67890)

tada!
""".lstrip()


@pytest.fixture
def example(tmp_path):
    """Write the EXAMPLE to a python file and return that
    """
    source = tmp_path / "example.py"
    source.write_text(EXAMPLE)
    return source


def test_ex2rst(example):
    """Test that `ex2rst` does what we think it should
    """
    # render the example to RST with ex2rst
    rst = example.with_suffix(".rst")
    ex2rst.main([str(example), str(rst)])

    # check that it matches
    assert rst.read_text() == EXAMPLE_OUTPUT
