# Copyright (c) 2021-2025 Cardiff University
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

"""Error handling for the GWpy test suite."""

from __future__ import annotations

import socket
from functools import wraps
from ssl import SSLError
from typing import TYPE_CHECKING
from urllib.error import URLError

import pytest
import requests.exceptions
from igwn_segments import segment

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        ParamSpec,
        TypeVar,
    )

    P = ParamSpec("P")
    R = TypeVar("R")

# -- Network/HTTP --------------------

SERVER_ERROR = segment(500, 600)

NETWORK_ERROR: tuple[type[Exception], ...] = (
    ConnectionError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    socket.timeout,
    SSLError,
    TimeoutError,
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
        f"failed to import exception types from pytest_socket: {exc}",
        stacklevel=2,
    )
else:
    NETWORK_ERROR = (
        *NETWORK_ERROR,
        SocketBlockedError,
        SocketConnectBlockedError,
    )


pytest_rerun_flaky_httperror = pytest.mark.flaky(
    reruns=2,
    reruns_delay=5,
    only_rerun=["HTTPError"],
)


def pytest_skip_network_error(
    func: Callable[P, R],
) -> Callable[P, R]:
    """Execute `func` but skip if it raises one of the network exceptions."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except* NETWORK_ERROR as exc:  # pragma: no cover
            if not exc.message and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]  # type: ignore[assignment]
            pytest.skip(str(exc))
        except* requests.exceptions.HTTPError as exc:  # pragma: no cover
            for error in exc.exceptions:
                if (
                    isinstance(error, requests.exceptions.HTTPError)
                    and error.response.status_code not in SERVER_ERROR
                ):
                    raise
            pytest.skip(str(exc.exceptions[0]))

    return wrapper


def pytest_skip_flaky_network(
    func: Callable,
) -> Callable:
    """Decorate test ``func`` with all necessary network decorators.

    A test decorated with this decorator will attempt behave as follows

    - if an `HTTPError` is encountered, retry a couple of times with
      a 5-second delay, then 'skip' if the response is a 5xx code,
      otherwise fail,
    - if any non-HTTP network-related error is encountered, skip
      the test.
    """
    for dec in (
        pytest_skip_network_error,
        pytest_rerun_flaky_httperror,
    ):
        func = dec(func)
    return func
