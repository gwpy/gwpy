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

"""Utilities for unified input/output."""

from __future__ import annotations

import gzip
import os
import tempfile
import warnings
from functools import wraps
from io import IOBase
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        ParamSpec,
        TypeAlias,
    )

    from lal.utils import CacheEntry

    # Type variables for decorator typing
    P = ParamSpec("P")
    T = TypeVar("T")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "FILE_LIKE",
    "FileLike",
    "NamedFileLike",
    "NamedReadable",
    "NamedWritable",
    "Readable",
    "Writable",
    "file_list",
    "file_path",
    "gopen",
    "with_open",
]


# -- IO type hints -------------------

AnyStr = TypeVar("AnyStr", str, bytes)

@runtime_checkable
class NamedIO(Protocol[AnyStr]):
    """Typing protocol for file-like objects with a name."""

    name: str

    # Core IO methods from typing.IO[str]
    def close(self) -> None:
        """Close the file."""
    def flush(self) -> None:
        """Flush the file."""
    def isatty(self) -> bool:
        """Return True if the file is connected to a terminal."""
    def read(self, n: int = -1) -> AnyStr:
        """Read n bytes from the file."""
    def readable(self) -> bool:
        """Return True if the file is readable."""
    def readline(self, limit: int = -1) -> AnyStr:
        """Read a single line from the file."""
    def readlines(self, hint: int = -1) -> list[AnyStr]:
        """Read all lines from the file."""
    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to a position in the file."""
    def seekable(self) -> bool:
        """Return True if the file supports seeking."""
    def tell(self) -> int:
        """Return the current position in the file."""
    def truncate(self, size: int | None = None) -> int:
        """Truncate the file to a given size."""
    def writable(self) -> bool:
        """Return True if the file is writable."""
    def write(self, s: AnyStr) -> int:
        """Write data to the file."""
    def writelines(self, lines: list[AnyStr]) -> None:
        """Write a list of lines to the file."""

    # Context manager
    def __enter__(self) -> NamedIO[AnyStr]:
        """Enter the runtime context related to this object."""
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit the runtime context related to this object."""

    # Iterator protocol
    def __iter__(self) -> NamedIO[AnyStr]:
        """Return an iterator over the lines of the file."""
    def __next__(self) -> AnyStr:
        """Return the next line from the file."""


#: Type alias for file-like objects
FileLike: TypeAlias = IOBase | gzip.GzipFile | tempfile._TemporaryFileWrapper  # noqa: SLF001

#: Type alias for file-like objects with a ``name`` attribute
NamedFileLike: TypeAlias = NamedIO | gzip.GzipFile | tempfile._TemporaryFileWrapper  # noqa: SLF001

#: Type alias for file system paths (that can be opened)
FileSystemPath: TypeAlias = str | os.PathLike

#: Type alias for readable objects (path or file-like)
Readable: TypeAlias = FileSystemPath | FileLike

#: Type alias for writable objects (path or file-like)
Writable = Readable

#: Type alias for named readable objects (path or named file-like)
NamedReadable: TypeAlias = FileSystemPath | NamedFileLike

#: Type alias for named writable objects (path or named file-like)
NamedWritable = NamedReadable

#: Tuple of file-like types
#:
#: .. deprecated:: 4.0.0
#:
#:     This type tuple is deprecated, use `FileLike` instead
FILE_LIKE = (
    IOBase,
    gzip.GzipFile,
    tempfile._TemporaryFileWrapper,  # noqa: SLF001
)


# -- File open utilities -------------

@overload
def with_open(
    func: Callable[P, T],
    mode: str = "r",
    pos: int = 0,
) -> Callable[P, T]: ...

@overload
def with_open(
    func: None = None,
    mode: str = "r",
    pos: int = 0,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

def with_open(
    func: Callable[P, T] | None = None,
    mode: str = "r",
    pos: int = 0,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a function to ensure the chosen argument is an open file.

    Parameters
    ----------
    func : `callable`
        the function to decorate

    mode : `str`, optional
        the mode with which to open files

    pos : `int`, optional
        which argument to look at

    Examples
    --------
    To ensure that the first argument is an open read-only file, just use
    the decorator without functional parentheses or arguments:

    >>> @with_open
    >>> def my_func(pathorfile, *args, **kwargs)
    >>>     ...

    To ensure that the second argument (position 1) is a file open for writing:

    >>> @with_open(mode="w", pos=1)
    >>> def my_func(stuff, pathorfile, *args, **kwargs)
    >>>     stuff.write_to(pathorfile, *args, **kwargs)
    """
    def _decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> T:
            # if the relevant positional argument isn't an open
            # file, or something that looks like one, ...
            source = args[pos]
            if not isinstance(source, FileLike):
                source = cast("FileSystemPath", source)
                # open the file, ...
                with open(source, mode=mode) as fobj:  # noqa: PTH123
                    # replace the argument with the open file, ...
                    args = list(args)  # type: ignore[assignment]
                    args[pos] = fobj  # type: ignore[index]
                    # and re-execute the function call
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapped_func
    if func:
        return _decorator(func)
    return _decorator


# -- file list utilities -------------

def file_list(
    flist: NamedReadable | list[NamedReadable] | tuple[NamedReadable, ...],
) -> list[str]:
    """Parse a number of possible input types into a list of filepaths.

    Parameters
    ----------
    flist : `file-like` or `list-like` iterable
        The input data container, normally just a single file path, or a list
        of paths, but can generally be any of the following

        - `str` representing a single file path (or comma-separated collection)
        - open `file` or `~gzip.GzipFile` object
        - :class:`~lal.utils.CacheEntry`
        - `str` with ``.cache`` or ``.lcf`` extension
        - simple `list` or `tuple` of `str` paths

    Returns
    -------
    files : `list`
        `list` of `str` file paths

    Raises
    ------
    ValueError
        if the input `flist` cannot be interpreted as any of the above inputs
    """
    # open a cache file and return list of paths
    if (
        isinstance(flist, str)
        and flist.endswith((".cache", ".lcf", ".ffl"))
    ):
        from .cache import read_cache
        return read_cache(flist)

    # separate comma-separate list of names
    if isinstance(flist, str):
        return flist.split(",")

    # parse list of entries (of some format)
    if isinstance(flist, list | tuple):
        return list(map(file_path, flist))

    # otherwise parse a single entry
    try:
        return [file_path(flist)]
    except ValueError as exc:
        exc.args = (
            f"Could not parse input {flist!r} as "
            "one or more file-like objects",
        )
        raise


def file_path(fobj: NamedReadable | bytes | CacheEntry) -> str:
    """Determine the path of a file.

    This doesn't do any sanity checking to check that the file
    actually exists, or is readable.

    Parameters
    ----------
    fobj : `file`, `str`, `os.PathLike`, `bytes`, `CacheEntry`, ...
        The file object or path to parse.

    Returns
    -------
    path : `str`
        The path of the underlying file, always as a `str`.

    Raises
    ------
    ValueError
        If a file path cannnot be determined.

    Examples
    --------
    >>> from gwpy.io.utils import file_path
    >>> import pathlib
    >>> file_path("test.txt")
    'test.txt'
    >>> file_path(pathlib.Path('dir') / 'test.txt')
    'dir/test.txt'
    >>> file_path(open("test.txt", "r"))
    'test.txt'
    >>> file_path("file:///home/user/test.txt")
    '/home/user/test.txt'
    """
    if isinstance(fobj, bytes):
        fobj = fobj.decode("utf-8")
    # file:// URL
    if isinstance(fobj, str) and fobj.startswith("file:"):
        return urlparse(fobj).path
    # Path-like object
    if isinstance(fobj, FileSystemPath):
        return os.fspath(fobj)
    # Named file-like object
    if isinstance(fobj, FileLike) and hasattr(fobj, "name"):
        return fobj.name
    # CacheEntry (or any other object with a .path attribute)
    if hasattr(fobj, "path"):
        return os.fspath(fobj.path)
    # Cannot parse
    msg = f"cannot parse file name for {fobj!r}"
    raise ValueError(msg)


# -- deprecated Gzip support ---------

GZIP_SIGNATURE = b"\x1f\x8b\x08"


def gopen(name, *args, **kwargs):
    """Open a file handling optional gzipping.

    .. deprecated:: 4.0.0

        This function is deprecated and will be removed in a future release.

    If ``name`` ends with ``'.gz'``, or if the GZIP file signature is
    found at the beginning of the file, the file will be opened with
    `gzip.open`, otherwise a regular file will be returned from `open`.

    Parameters
    ----------
    name : `str`, `pathlib.Path`
        path or name of file to open.

    *args, **kwargs
        other arguments to pass to either `open` for regular files, or
        `gzip.open` for gzipped files.

    Returns
    -------
    file : `io.TextIoBase`, `file`, `gzip.GzipFile`
        the open file object
    """
    warnings.warn(
        "gwpy.io.utils.gopen is deprecated and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

    # filename declares gzip
    if str(name).endswith(".gz"):
        return gzip.open(name, *args, **kwargs)

    # open regular file
    fobj = open(name, *args, **kwargs)
    sig = fobj.read(3)
    fobj.seek(0)
    if sig == GZIP_SIGNATURE:  # file signature declares gzip
        fobj.close()  # GzipFile won't close orig file when it closes
        return gzip.open(name, *args, **kwargs)
    return fobj
