# Copyright (c) 2017-2025 Cardiff University
#               2014-2017 Louisiana State University
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

"""A python package for gravitational-wave astrophysics.

GWpy is a collaboration-driven `Python <http://www.python.org>`_ package
providing tools for studying data from ground-based gravitational-wave
detectors.

GWpy provides a user-friendly, intuitive interface to the common time-domain
and frequency-domain data produced by the `LIGO <http://www.ligo.org>`_ and
`Virgo <http://www.ego-gw.it>`_ instruments and their analysis,
with easy-to-follow tutorials at each step.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "The LIGO Scientific Collaboration and the Virgo Collaboration"

import logging

from . import (
    log,
    plot,  # registers gwpy.plot.Axes as default rectilinear axes
)
from .utils.env import bool_env

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = ""


def init_logging(level: str | int = log.get_default_level()) -> None:
    """Quickly initialise logging for GWpy.

    This is mainly useful for scripts and interactive sessions to enable
    logging output from GWpy, and mainly for debugging purposes.

    For more control over logging, set up logging manually using the
    standard Python :mod:`logging` module, or see :ref:`gwpy-logging`.

    Parameters
    ----------
    level : int, optional
        The logging level to use.
        If not provided, the level from the ``GWPY_LOG_LEVEL`` environment
        variable will be used.
        If that variable is not set, ``INFO`` is used.

    Examples
    --------
    >>> import gwpy
    >>> gwpy.init_logging("DEBUG")
    """
    logger = log.init_logger(__name__, level=level or logging.INFO)
    logger.debug(
        "Initialised %s logging for %s",
        logging.getLevelName(logger.getEffectiveLevel()),
        logger.name,
    )


# Initialise logging for the package
if bool_env("GWPY_INIT_LOGGING", default=False):
    init_logging()
del bool_env
