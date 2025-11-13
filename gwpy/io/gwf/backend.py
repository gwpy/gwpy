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

"""GWF I/O backend discovery."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
    )
    from types import (
        FunctionType,
        ModuleType,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "BACKENDS",
    "get_backend",
    "get_backend_function",
    "import_backend",
]

# Preferentially ordered list of supported GWF backends
BACKENDS = [
    "frameCPP",
    "LALFrame",
]


def _backend_candidates(
    backends: list[str] = BACKENDS,
) -> Iterator[str]:
    default = os.getenv("GWPY_FRAME_LIBRARY") or backends[0]
    yield from dict.fromkeys([default, *backends])


def get_backend(
    package: str = __package__,
    backends: list[str] = BACKENDS,
) -> str:
    """Return the preferred GWF backend.

    This can be configured via the ``GWPY_FRAME_LIBRARY`` environment
    variable, which can be set to the name of any of the interface modules
    defined under `gwpy.io.gwf` (or the module path given by ``package``):

    - ``"frameCPP"``
    - ``"FrameL"``
    - ``"LALFrame"``

    Otherwise that list is manually searched in the order given above.

    Parameters
    ----------
    package : `str`
        The package under which to attempt to install a GWF backend module.

    backends : `list` of `str`
        The list of backends to try, defaults to the list of backends
        implemented in :mod:`gwpy.io.gwf.backend`.

    Examples
    --------
    If the environment variable ``GWPY_FRAME_LIBRARY`` is set:

    >>> os.environ["GWPY_FRAME_LIBRARY"] = "FrameL"
    >>> from gwpy.io.gwf import get_backend
    >>> get_backend()
    'FrameL'

    Or, if you have |LDAStools.frameCPP|_ installed:

    >>> from gwpy.io.gwf import get_backend
    >>> get_backend()
    'frameCPP'

    Or, if you don't have |lalframe|_:

    >>> get_backend()
    'LALFrame'

    Otherwise:

    >>> get_backend()
    ImportError: no GWF API available, please install a third-party GWF
    library (frameCPP, FrameL, LALFrame) and try again
    """
    for lib in _backend_candidates(backends):
        try:
            import_backend(lib.lower(), package=package)
        except ImportError:
            continue
        else:
            return lib
    msg = (
        "no GWF API available, please install a third-party GWF "
        f"library ({', '.join(BACKENDS)}) and try again"
    )
    raise ImportError(msg)


def import_backend(
    library: str,
    package: str = __package__,
) -> ModuleType:
    """Utility method to import the relevant GWF I/O backend.

    This is just a wrapper around :meth:`importlib.import_module` with
    a slightly nicer error message.
    """
    # import the frame library here to have any ImportErrors occur early
    try:
        return importlib.import_module(f".{library.lower()}", package=package)
    except ImportError as exc:
        exc.args = (
            f"Cannot import {library} GWF backend: {exc}",
        )
        raise


def get_backend_function(
    name: str,
    backend: str | None = None,
    backends: list[str] = BACKENDS,
    package: str = __package__,
) -> FunctionType:
    """Return the backend implementation of the function of the given name.

    Parameters
    ----------
    name : `str`
        The name of the function to import.

    backend : `str`
        The specific backend to use.

    backends : `list` of `str`
        The list of backends to query, defaults to all known backends,
        ignored if ``backend`` is given.

    package : `str`
        The package from which to attempt to install the backend module,
        defaults to `gwpy.io.gwf`.
    """
    backends_to_try: Iterable[str]
    if backend:
        backends_to_try = [backend]
    else:
        backends_to_try = _backend_candidates(backends)
    for bck in backends_to_try:
        try:
            mod = import_backend(bck, package=package)
            return getattr(mod, name)
        except ImportError:  # module not installed
            if backend:
                raise
            continue
        except AttributeError as exc:  # function not implemented
            if backend:  # user asked for this backend specifically
                msg = f"GWF backend '{backend}' does not implement '{name}'"
                raise NotImplementedError(msg) from exc
            continue
    # Try and import any backend to see if anything is actually available
    get_backend(package=package, backends=backends)
    # Otherwise we know that the requested function is not implemented
    # in any available backend
    msg = f"no GWF backend module found that implements '{name}'"
    raise NotImplementedError(msg)
