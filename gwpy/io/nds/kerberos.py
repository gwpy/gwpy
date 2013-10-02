
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
