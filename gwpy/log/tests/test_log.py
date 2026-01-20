# Copyright (c) 2025 Cardiff University
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

"""Tests for :mod:`gwpy.log`."""

import logging
import os
import sys
from importlib import reload
from unittest import mock

import pytest

from gwpy import log as gwpy_log


@mock.patch.dict("os.environ", clear=True)
@pytest.mark.parametrize(("env", "expected"), [
    pytest.param(None, logging.NOTSET, id="default"),
    pytest.param("DEBUG", logging.DEBUG, id="DEBUG"),
    pytest.param("debug", logging.DEBUG, id="debug"),
    pytest.param(10, logging.DEBUG, id="debug-int"),
    pytest.param("INFO", logging.INFO, id="INFO"),
    pytest.param(20, logging.INFO, id="info-int"),
    pytest.param("WARNING", logging.WARNING, id="WARNING"),
    pytest.param("ERROR", logging.ERROR, id="ERROR"),
    pytest.param("CRITICAL", logging.CRITICAL, id="CRITICAL"),
])
def test_get_default_level(env, expected):
    """Test :func:`gwpy.log.get_default_level`."""
    if env is not None:
        os.environ["GWPY_LOG_LEVEL"] = str(env)
    # Check that we get the expected level string
    assert gwpy_log.get_default_level() is expected


def test_init_logger():
    """Test :func:`gwpy.log.get_logger`."""
    # Basic test of getting a logger
    logger = gwpy_log.init_logger("gwpy.log")
    assert logger.name == "gwpy.log"
    # Check that we get the same logger if we call again
    assert gwpy_log.init_logger("gwpy.log") is logger


@mock.patch.dict("os.environ")
@pytest.mark.parametrize("env", [
    pytest.param({}, id="default"),
    pytest.param({
        "GWPY_LOG_LEVEL": "DEBUG",
    }, id="debug"),
    pytest.param({
        "GWPY_LOG_FORMAT": "%(name)s: %(message)s",
    }, id="format"),
    pytest.param({
        "GWPY_LOG_DATEFMT": "%H:%M:%S",
    }, id="datefmt"),
    pytest.param({
        "GWPY_LOG_LEVEL": "DEBUG",
        "GWPY_LOG_FORMAT": "%(name)s: %(message)s",
        "GWPY_LOG_DATEFMT": "%H:%M:%S",
    }, id="all"),
])
def test_init_logger_env(env):
    """Test :func:`gwpy.log.init_logger` with environment variables."""
    # Reset the test logger
    name = __name__
    logging.getLogger(name).handlers.clear()
    logging.getLogger(name).propagate = False

    # Update the env
    os.environ.update(env)

    # Reload the log module to reset any cached values
    reload(gwpy_log)

    # Check that the logger is initialised properly
    logger = gwpy_log.init_logger(name)
    assert logger.name == name

    if "GWPY_LOG_LEVEL" in env:
        assert logger.level == logging.DEBUG
    else:
        assert logger.level == logging.NOTSET

    handler = logger.handlers[0]
    assert (formatter := handler.formatter)
    assert formatter._fmt == os.getenv(
        "GWPY_LOG_FORMAT",
        gwpy_log.DEFAULT_LOG_FORMAT,
    )
    assert formatter.datefmt == os.getenv(
        "GWPY_LOG_DATEFMT",
        gwpy_log.DEFAULT_LOG_DATEFMT,
    )


# -- Test package-level log initialisation

@mock.patch.dict("sys.modules")
@mock.patch("gwpy.init_logging")
def test_package_init_logging(init_logging):
    """Test that we don't initialise logging at the package level by default."""
    del sys.modules["gwpy"]
    # By default the gwpy logger should not be configured.
    import gwpy  # noqa: F401
    init_logging.assert_not_called()


def test_package_init_logging_env():
    """Test that we can initialise logging at the package level."""
    del sys.modules["gwpy"]
    os.environ["GWPY_INIT_LOGGING"] = "1"
    # By default the gwpy logger should not be configured.
    import gwpy  # noqa: F401
    logger = logging.getLogger("gwpy")
    assert logger.level == logging.INFO
