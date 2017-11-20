# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016)
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

"""Extensions to `scipy.signal.signaltools`.
"""

from numpy import (asarray, reshape)

from scipy.signal._arraytools import (axis_slice, axis_reverse, odd_ext,
                                      even_ext, const_ext)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['sosfiltfilt']


def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=0):
    """Apply a digital filter forward and backward using second-order-sections
    """
    try:
        from scipy.signal.signaltools import (sosfilt, sosfilt_zi)
    except ImportError as exc:
        exc.args = ('{}, sosfiltfilt required scipy>=0.16',)
        raise

    x = asarray(x)

    # `method` is "pad"
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype "
                          "must be 'even', 'odd', 'constant', or None.") %
                         padtype)

    if padtype is None:
        padlen = 0
    if padlen is None:
        edge = sos.shape[0] * 6
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be at least "
                         "padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x

    # Get the steady state of the filter's step response.
    zi = sosfilt_zi(sos)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)
    zix0 = reshape(zi * x0, (sos.shape[0], 2))

    # Forward filter
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zix0)

    # Backward filter
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    ziy0 = reshape(zi * y0, (sos.shape[0], 2))

    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=ziy0)

    # Reverse y
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y
