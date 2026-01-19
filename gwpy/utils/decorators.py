# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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
from typing import (
    TYPE_CHECKING,
    no_type_check,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        ParamSpec,
        TypeVar,
    )

    P = ParamSpec("P")
    R = TypeVar("R")
    ReturnType = TypeVar("ReturnType")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

DEPRECATED_FUNCTION_WARNING: str = (
    "{0.__module__}.{0.__name__} has been deprecated, and will be "
    "removed in a future release."
)


class deprecated_property(property):  # noqa: N801
    """Sub-class of `property` that invokes DeprecationWarning on every call.

    .. deprecated:: 4.0.0

        This property is deprecated and will be removed in a future release.
        Use `warnings.deprecated` instead.
    """

    @no_type_check
    def __init__(
        self,
        fget: Callable,
        fset: Callable | None = None,
        fdel: Callable | None = None,
        doc: str | None = None,
    ) -> None:
        """Create a property that will issue a DeprecationWarning."""
        warnings.warn(
            "the deprecated_property decorator is itself deprecated and will be "
            "removed in a future release, please use the warnings.deprecated decorator "
            "from the Python 3.13+ standard library instead",
            category=DeprecationWarning,
            stacklevel=2,
        )

        # get name of property
        pname = fget.__name__

        # build a wrapper that will spawn a DeprecationWarning for all calls
        def _warn(func: Callable[P, R]) -> Callable[P, R]:
            """Wrap a function to issue a DeprecationWarning."""
            @wraps(func)
            def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
                """Wrap a function to issue a DeprecationWarning."""
                inst = args[0]
                warnings.warn(
                    f"the {type(inst).__name__}.{pname} property is deprecated "
                    "and will be removed in a future release, please stop "
                    "using it.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

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
    """Add a `DeprecationWarning` to a function.

    .. deprecated:: 4.0.0

        This function is deprecated and will be removed in a future release.
        Use `warnings.deprecated` instead.

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
    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> R:
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


def return_as(
    returntype: Callable[[R], ReturnType],
) -> Callable[[Callable[P, R]], Callable[P, ReturnType]]:
    """Decorate a function to cast the return as the given type.

    Parameters
    ----------
    returntype : `type`
        the desired return type of the decorated function
    """
    def decorator(func: Callable[P, R]) -> Callable[P, ReturnType]:
        """Decorate a function to cast the return value as the given type."""
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> ReturnType:
            """Wrap a function to cast the return value as the given type."""
            result = func(*args, **kwargs)
            try:
                return returntype(result)
            except (TypeError, ValueError) as exc:
                fname = getattr(func, "__name__", str(func))
                rname = getattr(returntype, "__name__", str(returntype))
                exc.args = (
                    f"failed to cast return from {fname} as {rname}: {exc}",
                )
                raise

        return wrapped

    return decorator
