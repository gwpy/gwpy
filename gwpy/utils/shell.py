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

"""Utilities for calling out to the shell
"""

import os
from subprocess import (Popen, PIPE, CalledProcessError)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


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
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    fpath = os.path.split(program)[0]
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    raise ValueError("No executable '%s' found in PATH" % program)


def call(cmd, stdout=PIPE, stderr=PIPE, on_error='raise'):
    """Call out to the shell using `subprocess.Popen`
    """
    proc = Popen(cmd, stdout=stdout, stderr=stderr)
    out, err = proc.communicate()
    if proc.returncode:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            e = CalledProcessError(proc.returncode, cmd=' '.join(frchannels),
                                   output=err)
            warnings.warn(str(e))
        else:
            raise CalledProcessError(proc.returncode, cmd=' '.join(frchannels),
                                     output=err)
    return out, err
