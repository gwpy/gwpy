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

"""Utility module to initialise a kerberos ticket for NDS2 connections
"""

import getpass
import os

try:
    raw_input
except NameError:
    raw_input = input

from subprocess import (PIPE, Popen)

from ... import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ['kinit']


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
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    raise ValueError("No executable '%s' found in PATH" % program)


def kinit(username=None, password=None, realm='LIGO.ORG', exe=None):
    """Initialise a kerboeros ticket to enable authenticated connections
    to the NDS2 server network
    """
    if exe is None:
        exe = which('kinit')
    if username is None:
        username = raw_input("Please provide username for the %s kerberos "
                             "realm: " % realm)
    if password is None:
        password = getpass.getpass(prompt="Password for %s@%s: "
                                          % (username, realm))
    kget = Popen([exe, '%s@%s' % (username, realm)], stdout=PIPE,
                 stderr=PIPE, stdin=PIPE)
    kget.stdin.write('%s\n' % password)
    kget.wait()
    print("Kerberos ticket generated for %s@%s" % (username, realm))
