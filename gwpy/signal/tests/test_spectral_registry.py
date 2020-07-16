# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Tests for :mod:`gwpy.signal.spectral.registry`
"""

import pytest

from ..spectral import _registry as fft_registry


def teardown_module():
    # remove test methods from registry
    # otherwise they will impact other tests, and test ordering
    # is annoying to predict
    fft_registry.METHODS.pop('fake_method', '')


def test_register_get():
    """Test :mod:`gwpy.signal.spectral.registry`
    """
    def fake_method():
        pass

    # test register
    fft_registry.register_method(fake_method)
    assert 'fake_method' in fft_registry.METHODS

    # test get
    f = fft_registry.get_method('fake_method')
    assert f is fake_method

    with pytest.raises(KeyError):
        fft_registry.get_method('unregistered')
