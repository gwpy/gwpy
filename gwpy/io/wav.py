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

"""Write a `TimeSeries` to a WAV file

"""

from numpy import (savetxt, loadtxt)

from scipy.io import wavfile

from .registry import (register_reader, register_writer, register_identifier)
from .utils import identify_factory

__author__ = "Ryan Fisher <ryan.fisher@ligo.org>"


#def read_ascii(filepath, _obj=Series, xcol=0, ycol=1, delimiter=None,
#               **kwargs):
#    """Read a `Series` from an ASCII file
#    """
#    # get specific args for loadtxt
#    loadargs = {'unpack': True, 'usecols': [xcol, ycol]}
#    for kwarg in ['dtype', 'comments', 'delimiter', 'converters', 'skiprows']:
#        if kwarg in kwargs:
#            loadargs[kwarg] = kwargs.pop(kwarg)
#    # read data, format and return
#    x, y = loadtxt(filepath, delimiter=delimiter, **loadargs)
#    return _obj(y, xindex=x, **kwargs)


def write_wav(series,fobj,rate=4096,amp=.1):
    """Prepares the timeseries for audio and writes 
    to a .wav file.
    
    Parameters
    ----------
    series : :class:`~gwpy.data.TimeSeries`
        data series to write
    fobj : `str`, `file`
        file object, or path to file, to write to
    
    rate: `float`, optional, default=4096
        rate in Hz of the .wav file.
    amp: `float`, optional, default=.1
        maximum amplitude of .wav file.

    See Also
    --------
    scipy.io.wavfile.write
        for details on the write process. 
    """

    audio_resamp = series.resample(rate)
    audio_normal  = amp * audio_resamp.value / (max(abs(audio_resamp.value)))

    wavfile.write(fobj,rate,audio_normal)


###def write_ascii(series, fobj, fmt='%.18e', delimiter=' ', newline='\n',
#                header='', footer='', comments='# '):
#    """Write a `Series` to a file in WAV format
#
#    Parameters
#    ----------
#    series : :class:`~gwpy.data.Series`
#        data series to write
#    fobj : `str`, `file`
#        file object, or path to file, to write to
#
#    See also
#    --------
#    numpy.: for documentation of keyword arguments
#    """
#    x = series.xindex.value
#    y = series.value
#    return savetxt(fobj, zip(x, y), fmt=fmt, delimiter=delimiter,
#                   newline=newline, header=header, footer=footer,
#                   comments=comments)


#formats = {'txt': None,
#           'csv': ','}
formats = ['wav']


def wav_io_factory(obj, delimiter=None):
    """Probably unneeded for this simple class
    """
    #def _read(filepath, **kwargs):
    #    kwargs.setdefault('delimiter', delimiter)
    #    return read_wav(filepath, _obj=obj, **kwargs)

    def _write(series, filepath, **kwargs):
        #kwargs.setdefault('delimiter', delimiter or ' ')
        return write_wav(series, filepath, **kwargs)
    #return _read, _write
    return _write


def register_wav(obj):
    """Register WAV I/O methods for given type obj

    This factory method registers 'wav' I/O format with
    a writer and auto-identifier
    """
    #for form, delim in formats.iteritems():
    for form in formats: # kept this loop incase we can support mp3 eventually
        write_ = wav_io_factory(obj, **kwargs)
        register_identifier('wav', obj, identify_factory(form, '%s.gz' % form))
        register_writer(form, obj, write_)
        #register_reader(form, obj, read_)
