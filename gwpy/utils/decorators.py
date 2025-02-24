# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

"""Decorators for GWpy."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Callable

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

DEPRECATED_FUNCTION_WARNING: str = (
    "{0.__module__}.{0.__name__} has been deprecated, and will be "
    "removed in a future release."
)


class deprecated_property(property):  # noqa: N801
    """Sub-class of `property` that invokes DeprecationWarning on every call."""
    def __init__(
        self,
        fget: Callable,
        fset: Callable | None = None,
        fdel: Callable | None = None,
        doc: str | None = None,
    ):
        # get name  of property
        pname = fget.__name__

        # build a wrapper that will spawn a DeprecationWarning for all calls
        def _warn(func):
            @wraps(func)
            def _wrapped(self, *args, **kwargs):
                warnings.warn(
                    f"the {type(self).__name__}.{pname} property is deprecated "
                    "and will be removed in a future release, please stop "
                    "using it.",
                    DeprecationWarning,
                )
                return func(self, *args, **kwargs)

            return _wrapped

        # wrap the property methods
        if fdel:
            fdel = _warn(fdel)
        if fset:
            fset = _warn(fset)
        if not fset and not fdel:  # only wrap once
            fget = _warn(fget)

        super().__init__(fget, fset, fdel, doc)


def deprecated_function(
    func: Callable | None = None,
    message: str = DEPRECATED_FUNCTION_WARNING,
) -> Callable:
    """Adds a `DeprecationWarning` to a function.

    Parameters
    ----------
    func : `callable`
        The function to decorate with a `DeprecationWarning`.

    message : `str`, optional
        The warning message to present.

    Notes
    -----
    The final warning message is formatted using `str.format`,
    e.g ``message.format(func)``, so you can use attribute references
    to the function itself.
    See the default message as an example.
    """
    def _decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            warnings.warn(
                message.format(func),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapped_func
    if func:
        return _decorator(func)
    return _decorator


def return_as(returntype: type) -> Callable:
    """Decorator to cast return of function as the given type.

    Parameters
    ----------
    returntype : `type`
        the desired return type of the decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                return returntype(result)
            except (TypeError, ValueError) as exc:
                exc.args = (
                    f"failed to cast return from {func.__name__} as "
                    f"{returntype.__name__}: {exc}",
                )
                raise
        return wrapped

    return decorator
