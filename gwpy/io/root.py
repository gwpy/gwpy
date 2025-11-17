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

"""I/O utilities for ROOT files."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    from .utils import (
        FileLike,
        FileSystemPath,
    )

__all__ = [
    "identify_root",
]

ROOT_SIGNATURE = b"root"
ROOT_SIGNATURE_T = ROOT_SIGNATURE.decode("utf-8")


def identify_root(
    origin: Literal["read", "write"],  # noqa: ARG001
    filepath: FileSystemPath | None,
    fileobj: FileLike | None,
    *args,  # noqa: ANN002
    **kwargs,  # noqa: ARG001
) -> bool:
    """Identify a filename or file object as ROOT.

    This function is only indented to be used as a registered
    identifier for Astropy's Unified I/O system.
    """
    # try and read file descriptor
    if fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            signature = fileobj.read(4)
        finally:
            fileobj.seek(loc)
        return signature in (ROOT_SIGNATURE, ROOT_SIGNATURE_T)
    # otherwise guess from file name
    if filepath is not None:
        return str(filepath).endswith(".root")
    # or finally introspect an object from uproot
    try:
        import uproot
    except ImportError:
        return False
    return isinstance(args[0], uproot.ReadOnlyFile | uproot.ReadOnlyDirectory)
