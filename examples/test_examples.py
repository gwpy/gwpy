# Copyright (c) 2019-2025 Cardiff University
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

"""Test suite for all examples."""

import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import pytest
from matplotlib import use

from gwpy.io.nds2 import NDSWarning
from gwpy.testing.errors import pytest_skip_flaky_network

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

use("agg")  # force non-interactive backend

# find all examples
EXAMPLE_BASE = Path(__file__).parent
EXAMPLE_DIRS = [exdir for exdir in EXAMPLE_BASE.iterdir() if exdir.is_dir()]
EXAMPLES = sorted([
    pytest.param(expy, id=str(expy.relative_to(EXAMPLE_BASE))) for
    exdir in EXAMPLE_DIRS for expy in exdir.glob("*.py")],
)


@contextmanager
def cwd(path):
    """Context manager to change to a given directory and back again."""
    oldpwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(oldpwd))


# -- fixtures ---------------

@pytest.fixture(autouse=True)
def close_figures():
    """Fixture to close all matplotlib figures after each test."""
    from matplotlib import pyplot
    try:
        yield
    finally:
        # close all open figures regardless of test status
        pyplot.close("all")


# acceptable authentication failures
NDS2_AUTH_FAILURES = [
    "failed to establish a connection",
    "error authenticating against",
]
NDS2_SKIP = re.compile(
    "({})".format("|".join(NDS2_AUTH_FAILURES)),
    re.IGNORECASE,
)

# stderr message for NDS2 data access failure
NDS2_READ_ERROR_STDERR = re.compile(r"read_server_response: Wrong length read")


def skip_nds_authentication_error(func):
    """Ignore NDS2 authentication errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            with warnings.catch_warnings():  # force NDS warnings as errors
                for msg in NDS2_AUTH_FAILURES:
                    warnings.filterwarnings(
                        "error",
                        message=msg,
                        category=NDSWarning,
                    )
                return func(*args, **kwargs)  # run the test
        except NDSWarning as exc:  # pragma: no-cover
            # if we can't authenticate, dont worry
            if NDS2_SKIP.match(str(exc)):
                pytest.skip(str(exc))
            raise

    return wrapper


def skip_missing_optional_dependency(func):
    """Ignore missing optional dependencies (mainly in `TimeSeries.get`)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)  # run the test
        except (
            ImportError,  # direct import error
            RuntimeError,  # chained error
        ) as exc:
            msg = str(exc)
            cause = exc.__cause__
            while cause:  # walk up the error chain
                msg += f" caused by '{cause}'"
                cause = cause.__cause__
            # needs an optional dependency
            if not isinstance(exc, ImportError):
                raise
            pytest.skip(msg)

    return wrapper


# -- test -------------------

@pytest.mark.parametrize("script", EXAMPLES)
@pytest.mark.filterwarnings(
    "ignore:Matplotlib is currently using agg",
    "ignore:.*Agg is non-interactive",
)
@pytest_skip_flaky_network  # if there are network errors, we don't care
@skip_nds_authentication_error
@skip_missing_optional_dependency
def test_example(capfd, script):
    """Test an example script runs without error."""
    # read example code from file
    code = compile(
        script.read_text(),
        str(script),
        "exec",
    )

    # in the directory of the script, run it
    try:
        with cwd(script.parent):
            exec(code, globals())
    except* RuntimeError as exc:  # NDS2 failed
        stderr = capfd.readouterr().err
        if NDS2_READ_ERROR_STDERR.match(stderr):
            pytest.skip(f"{exc} caused by '{NDS2_READ_ERROR_STDERR.pattern}'")
        raise
