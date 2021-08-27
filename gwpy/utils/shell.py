# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Utilities for calling out to the shell
"""

import warnings
from distutils.spawn import find_executable
from subprocess import (Popen, PIPE, CalledProcessError)

from .decorators import deprecated_function

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@deprecated_function(
    message=(
        "gwpy.utils.shell.which is deprecated in favour of shutil.which from "
        "the Python standard library, and will be removed in a future release"
    ),
)
def which(program):
    """Find full path of executable program

    Parameters
    ----------
    program : `str`
        path of executable name for which to search

    Returns
    -------
    programpath
        the full absolute path of the executable

    Raises
    ------
    ValueError
        if not executable program is found
    """
    exe = find_executable(program)
    if exe is None:
        raise ValueError("No executable '%s' found in PATH" % program)
    return exe


@deprecated_function(
    message=(
        "gwpy.utils.shell.call is deprecated in favour of subprocess.call "
        "from the Python standard library, and will be removed in a "
        "future release."
    ),
)
def call(cmd, stdout=PIPE, stderr=PIPE, on_error='raise', **kwargs):
    """Call out to the shell using `subprocess.Popen`

    Parameters
    ----------
    stdout : `file-like`, optional
        stream for stdout

    stderr : `file-like`, optional
        stderr for stderr

    on_error : `str`, optional
        what to do when the command fails, one of

        - 'ignore' - do nothing
        - 'warn' - print a warning
        - 'raise' - raise an exception

    **kwargs
        other keyword arguments to pass to `subprocess.Popen`

    Returns
    -------
    out : `str`
        the output stream of the command
    err : `str`
        the error stream from the command

    Raises
    ------
    OSError
        if `cmd` is a `str` (or `shell=True` is passed) and the executable
        is not found
    subprocess.CalledProcessError
        if the command fails otherwise
    """
    if isinstance(cmd, (list, tuple)):
        cmdstr = ' '.join(cmd)
        kwargs.setdefault('shell', False)
    else:
        cmdstr = str(cmd)
        kwargs.setdefault('shell', True)
    proc = Popen(cmd, stdout=stdout, stderr=stderr, **kwargs)
    out, err = proc.communicate()
    if proc.returncode:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            e = CalledProcessError(proc.returncode, cmdstr)
            warnings.warn(str(e))
        else:
            raise CalledProcessError(proc.returncode, cmdstr)
    return out.decode('utf-8'), err.decode('utf-8')
