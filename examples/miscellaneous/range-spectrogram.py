#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2019-2020)
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

"""Estimating the spectral contribution to inspiral range

We have seen how the binary neutron star (BNS) inspiral range of a
gravitational-wave detector can be measured directly from the strain
readout. In this example, we will estimate the average spectral
contribution to BNS range from the strain record surrounding GW170817
using :func:`gwpy.astro.range_spectrogram`.
"""

__author__ = 'Alex Urban <alexander.urban@ligo.org>'

# First, we need to load some data. As before we can `fetch` the
# `public data <https://www.gw-openscience.org/catalog/>`__
# around the GW170817 BNS merger:

from gwpy.timeseries import TimeSeries
l1 = TimeSeries.fetch_open_data('L1', 1187006834, 1187010930)

# Then, we can calculate a `Spectrogram` of the inspiral range
# amplitude spectrum:

from gwpy.astro import range_spectrogram
l1spec = range_spectrogram(l1, 30, fftlength=4, fmin=15, fmax=500) ** (1./2)

# We can plot this `Spectrogram` to visualise spectral variation in
# LIGO-Livingston's sensitivity in the hour or so surrounding GW170817:

plot = l1spec.plot(figsize=(12, 5))
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(15, 500)
ax.set_title('LIGO-Livingston sensitivity to BNS around GW170817')
ax.set_epoch(1187008882)  # <- set 0 on plot to GW170817
ax.colorbar(cmap='cividis', clim=(0, 16),
            label='BNS range amplitude spectral density '
                  r'[Mpc/$\sqrt{\mathrm{Hz}}$]')
plot.show()

# Note, the extreme dip in sensitivity near GW170817 is caused by a
# loud, transient noise event, see `Phys. Rev. Lett. vol. 119, p.
# 161101 <http://doi.org/10.1103/PhysRevLett.119.161101>`_ for more
# information.
