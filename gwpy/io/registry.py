# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

"""I/O registry extensions on top of `astropy.io.registry`

This module imports a subset of the useful functions from
:mod:`astropy.io.registry` for convenience.
"""

import sys
from functools import wraps

from astropy.io.registry import (  # noqa: F401
    get_reader,
    register_identifier as astropy_register_identifier,
    register_reader,
    register_writer,
)
try:
    from astropy.io.registry.compat import default_registry
except ModuleNotFoundError:  # astropy < 5
    from astropy.io.registry import _get_valid_format as get_format
else:
    get_format = default_registry._get_valid_format

from astropy.utils.data import get_readable_fileobj

from .utils import (file_list, FILE_LIKE)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def identify_with_list(identifier):
    """Decorate an I/O identifier to handle a list of files as input

    This function tries to resolve a single file path as a `str` from any
    file-like or collection-of-file-likes to pass to the underlying
    identifier for comparison.
    """
    @wraps(identifier)
    def decorated_func(origin, filepath, fileobj, *args, **kwargs):
        # pylint: disable=missing-docstring
        try:
            filepath = file_list(filepath)[0]
        except ValueError:
            if filepath is None:
                try:
                    files = file_list(args[0])
                except (IndexError, ValueError):
                    pass
                else:
                    if files:
                        filepath = files[0]
        except IndexError:
            pass
        return identifier(origin, filepath, fileobj, *args, **kwargs)
    return decorated_func


def register_identifier(data_format, data_class, identifier, force=False):
    # pylint: disable=missing-docstring
    return astropy_register_identifier(
        data_format, data_class, identify_with_list(identifier), force=force)


register_identifier.__doc__ = astropy_register_identifier.__doc__


def get_read_format(cls, source, args, kwargs):
    """Determine the read format for a given input source
    """
    ctx = None
    if isinstance(source, FILE_LIKE):
        fileobj = source
        filepath = source.name if hasattr(source, 'name') else None
    else:
        filepath = source
        try:
            ctx = get_readable_fileobj(filepath, encoding='binary')
            fileobj = ctx.__enter__()  # pylint: disable=no-member
        except IOError:
            raise
        except Exception:  # pylint: disable=broad-except
            fileobj = None
    try:
        return get_format('read', cls, filepath, fileobj, args, kwargs)
    finally:
        if ctx is not None:
            ctx.__exit__(*sys.exc_info())  # pylint: disable=no-member
