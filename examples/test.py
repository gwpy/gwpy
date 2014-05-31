#!/usr/bin/env python

import numpy
from gwpy.plotter import figure

fig = figure()
ax = fig.gca()
ax.plot(1000000000 + numpy.arange(1440) * 60, numpy.random.random(1440))
ax.set_xscale('auto-gps')
fig.save('test.png')
fig.close()
