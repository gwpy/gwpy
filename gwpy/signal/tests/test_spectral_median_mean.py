# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Unit test for :mod:`gwpy.signal.spectral._median_mean`
"""

from unittest import mock

import pytest

from ..spectral import _median_mean as fft_median_mean


@pytest.mark.requires("lal", "pycbc.psd")
@mock.patch(
    "gwpy.signal.spectral._pycbc.welch",
    side_effect=(None, ImportError, ImportError),
)
@mock.patch(
    "gwpy.signal.spectral._lal._lal_spectrum",
    side_effect=(None, ImportError),
)
def test_median_mean(lal_func, pycbc_func):
    """Check that the registered "median-mean" method works

    Should resolve in this order to

    - ``pycbc_median_mean``
    - ``lal_median_mean``
    - `KeyError`
    """
    # first call goes to pycbc
    with pytest.deprecated_call() as record:
        fft_median_mean.median_mean(1, 2, 3)
    assert len(record) == 2  # once for pycbc, once for mm
    assert "pycbc_median_mean" in record[-1].message.args[0]
    pycbc_func.assert_called_once_with(1, 2, 3, avg_method="median-mean")

    # second call goes to lal
    with pytest.deprecated_call() as record:
        fft_median_mean.median_mean(1, 2, 3)
    assert len(record) == 3  # once for pycbc, once for lal, once for mm
    assert "lal_median_mean" in record[-1].message.args[0]
    lal_func.assert_called_once_with(
        1,
        2,
        noverlap=3,
        method='median-mean',
        window=None,
        plan=None,
    )

    # third call errors
    with pytest.deprecated_call(), pytest.raises(KeyError):
        fft_median_mean.median_mean(1, 2, 3)
