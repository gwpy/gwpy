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

"""I/O registry extensions on top of `astropy.io.registry`
"""

from functools import wraps

from astropy.io.registry import *

from .cache import file_list

# record astropy origin registry_identifier
astropy_register_identifier = register_identifier

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def identify_with_list(identifier):
    """Decorate an I/O identifier to handle a list of files as input

    This function tries to resolve a single file path as a `str` from any
    file-like or collection-of-file-likes to pass to the underlying
    identifier for comparison.
    """
    @wraps(identifier)
    def decorated_func(origin, filepath, fileobj, *args, **kwargs):
        try:
            filepath = file_list(filepath)[0]
        except ValueError:
            if filepath is None:
                try:
                    files = file_list(args[0])
                except (IndexError, ValueError):
                    pass
                else:
                    if len(files):
                        filepath = files[0]
        except IndexError:
            pass
        return identifier(origin, filepath, fileobj, *args, **kwargs)
    return decorated_func


def register_identifier(data_format, data_class, identifier, force=False):
    return astropy_register_identifier(
        data_format, data_class, identify_with_list(identifier), force=force)
register_identifier.__doc__ = astropy_register_identifier.__doc__
