# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2021)
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

"""Error handling for the GWpy test suite
"""

import socket
from functools import wraps
from ssl import SSLError
from urllib.error import URLError

import pytest

NETWORK_ERROR = (
    ConnectionError,
    socket.timeout,
    SSLError,
    URLError,
)

# attempt to also catch errors from pytest_socket,
# this should enable the test suite to run on machines that
# don't allow network access, e.g. debian build machines
try:
    from pytest_socket import (
        SocketBlockedError,
        SocketConnectBlockedError,
    )
except ModuleNotFoundError:  # pragma: no cover
    # pytest-socket not installed
    pass
except ImportError as exc:  # pragma: no cover
    # pytest-socket installed but errors not found,
    # print a warning to tell the devs to update this module
    import warnings
    warnings.warn(
        "failed to import exception types from pytest_socket: "
        "{}".format(str(exc)),
    )
else:
    NETWORK_ERROR = NETWORK_ERROR + (
        SocketBlockedError,
        SocketConnectBlockedError,
    )


def pytest_skip_network_error(func):
    """Execute `func` but skip if it raises one of the network exceptions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NETWORK_ERROR as exc:  # pragma: no cover
            pytest.skip(str(exc))

    return wrapper


def pytest_skip_cvmfs_read_error(func):
    """Execute `func` but skip if a CVMFS file fails to open

    This is most likely indicative of a broken CVMFS mount, which is not
    GWpy's problem.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:  # pragma: no cover
            # if function failed to read a CVMFS file, skip
            msg = str(exc)
            if msg.startswith("Unable to open file: /cvmfs"):
                pytest.skip(msg)
            # otherwise raise the original error
            raise

    return wrapper
