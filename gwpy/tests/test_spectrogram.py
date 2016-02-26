# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Unit test for spectrogram module
"""

from numpy import testing as nptest

from gwpy import version
from gwpy.spectrogram import Spectrogram

from test_array import Array2DTestCase

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


# -----------------------------------------------------------------------------

class SpectrogramTestCase(Array2DTestCase):
    """`~unittest.TestCase` for the `~gwpy.spectrogram.Spectrogram` class
    """
    TEST_CLASS = Spectrogram

    def test_epoch(self):
        array = self.create()
        self.assertEquals(array.epoch.gps, array.x0.value)

    def test_ratio(self):
        mean_ = self.TEST_ARRAY.ratio('mean')
        nptest.assert_array_equal(
            mean_.value,
            self.TEST_ARRAY.value / self.TEST_ARRAY.mean(axis=0).value)
        median_ = self.TEST_ARRAY.ratio('median')
        nptest.assert_array_equal(
            median_.value,
            self.TEST_ARRAY.value / self.TEST_ARRAY.median(axis=0).value)


if __name__ == '__main__':
    unittest.main()
