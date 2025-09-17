# Copyright (c) 2014-2017 Louisiana State University
#               2017-2021 Cardiff University
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

"""I/O registry extensions on top of `astropy.io.registry`.

This module imports a subset of the useful functions from
:mod:`astropy.io.registry` for convenience.
"""

from __future__ import annotations

import contextlib
import inspect
import warnings
from abc import (
    ABC,
    abstractmethod,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeVar,
    cast,
)

from astropy.io import registry as astropy_registry
from astropy.io.registry import (
    UnifiedReadWriteMethod,  # noqa: F401
)

from ..utils.env import bool_env
from ..utils.progress import progress_bar
from .remote import open_remote_file
from .utils import (
    file_list,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from typing import (
        BinaryIO,
        Literal,
    )

    from .utils import (
        FileLike,
        FileSystemPath,
        NamedReadable,
    )

# Type variable for Generic classes
T = TypeVar("T")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- Identify utilities --------------

class IdentifyProtocol(Protocol):
    """Typing protocol for Astropy I/O identify functions."""

    def __call__(
        self,
        origin: Literal["read", "write"],
        filepath: FileSystemPath | None,
        fileobj: FileLike | None,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> bool:
        """Identify the given extensions in a file object/path."""


def identify_factory(*extensions: str) -> IdentifyProtocol:
    """Return an Astropy Unified I/O identify function for a set of extensions.

    The returned function is designed for use in the unified I/O registry
    via the `astropy.io.registry.register_identifier` hook.

    Parameters
    ----------
    extensions : `str`
        one or more file extension strings

    Returns
    -------
    identifier : `callable`
        an identifier function that tests whether an incoming file path
        carries any of the given file extensions (using `str.endswith`)
    """
    def identify(
        origin: Literal["read", "write"],  # noqa: ARG001
        filepath: FileSystemPath | None,
        fileobj: FileLike | None,  # noqa: ARG001
        *args,  # noqa: ANN002, ARG001
        **kwargs,  # noqa: ARG001
    ) -> bool:
        """Identify the given extensions in a file object/path."""
        return (
            filepath is not None
            and str(filepath).endswith(extensions)
        )
    return identify


def list_identifier(
    identifier: IdentifyProtocol,
) -> IdentifyProtocol:
    """Decorate an I/O identifier to handle a list of files as input.

    This function tries to resolve a single file path as a `str` from any
    one or more file-like objects in the ``filepath`` or ``args`` inputs
    to pass to the underlying identifier for comparison.
    """
    def decorated_func(
        origin: Literal["read", "write"],
        filepath: FileSystemPath | None,
        fileobj: FileLike | None,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> bool:
        target = filepath  # thing to search
        if target is None and args:
            target = args[0]
        with contextlib.suppress(
            IndexError,  # empty list
            ValueError,  # target can't be resolved as a list of file-like
        ):
            filepath = file_list(target)[0]  # type: ignore[arg-type]
        return identifier(origin, filepath, fileobj, *args, **kwargs)

    return decorated_func


# -- Unified I/O format --------------

class UnifiedIORegistry(astropy_registry.UnifiedIORegistry):
    """UnifiedIORegistry that can handle reading files in parallel."""

    def identify_format(
        self,
        origin: Literal["read", "write"],
        data_class_required: type,
        path: FileSystemPath | None,
        fileobj: FileLike | None,
        args: tuple,
        kwargs: dict,
    ) -> list[str]:
        """Identify the format of a file, handling lists of files."""
        if fileobj is None:
            with contextlib.suppress(
                IndexError,  # list is empty
                ValueError,  # failed to parse as list-like
            ):
                path = file_list(path)[0]  # type: ignore[arg-type]
        return super().identify_format(
            origin,
            data_class_required,
            path,
            fileobj,
            args,
            kwargs,
        )

    identify_format.__doc__ = (
        astropy_registry.UnifiedIORegistry.identify_format.__doc__
    )

    def register_identifier(
        self,
        data_format: str,
        data_class: type,
        identifier: IdentifyProtocol,
        force: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """Register an identifier function that can handle lists of files."""
        return super().register_identifier(
            data_format,
            data_class,
            list_identifier(identifier),
            force=force,
        )

    register_identifier.__doc__ = (
        astropy_registry.UnifiedIORegistry.register_identifier.__doc__
    )


default_registry = UnifiedIORegistry()


class MergeProtocol(Protocol[T]):
    """Typing protocol for merge functions used in `UnifiedRead`."""

    def __call__(self, items: Sequence[T]) -> T:
        """Merge a sequence of items into a single item."""


class UnifiedRead(astropy_registry.UnifiedReadWrite, ABC, Generic[T]):
    """Base ``Class.read()`` implementation that handles parallel reads.

    Each parallel read must return an instance of the same type
    as the input class, and the results are merged by the provided
    ``merge_function`` which should have the following signature.

        def merge_function(self: Type, instances: list[Type], **kwargs) -> Type

    i.e take in the type object and a list of instances, and return
    a single instance of the same type.
    """

    def __init__(
        self,
        instance: object,
        cls: type[T],
        registry: UnifiedIORegistry = default_registry,
    ) -> None:
        """Initialise a new `UnifiedRead` instance."""
        super().__init__(
            instance,
            cls,
            "read",
            registry=registry,
        )

    # -- Merge -----------------------

    @abstractmethod
    def merge(self, items: Sequence[T], **kwargs) -> T:
        """Merge a sequence of items into a single item.

        Parameters
        ----------
        items : Sequence[T]
            The sequence of items to merge.
        **kwargs
            Additional keyword arguments specific to the merge operation.
            Subclasses may override this method with specific named parameters;
            use ``# type: ignore[override]`` to suppress mypy warnings about
            signature compatibility.

        Returns
        -------
        T
            The merged item.

        Notes
        -----
        The base class automatically extracts merge-specific kwargs from the
        ``__call__`` method by inspecting the merge method signature, so
        subclasses with specific named parameters will receive them correctly.
        """

    def _merge_kwargs(self, kwargs: dict, merge_function: MergeProtocol) -> dict:
        merge_sig = inspect.signature(merge_function)
        return {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in merge_sig.parameters and key not in ("self", "items")
        }

    # -- Read ------------------------

    def _read_single_file(
        self,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> T:
        """Execute ``cls.read()`` for a single file."""
        return self.registry.read(self._cls, *args, **kwargs)

    @staticmethod
    def _format_input_list(
        source: NamedReadable | list[NamedReadable],
    ) -> Sequence[NamedReadable]:
        """Format the input arguments to include a list of files."""
        # parse input as a list of files
        try:  # try and map to a list of file-like objects
            return file_list(source)
        except ValueError:  # otherwise treat as single file
            return [cast("NamedReadable", source)]

    def __call__(
        self,
        source: NamedReadable | list[NamedReadable],
        *args,  # noqa: ANN002
        format: str | None = None,  # noqa: A002
        cache: bool | None = None,
        parallel: int = 1,
        verbose: str | bool = False,
        merge_function: MergeProtocol | None = None,
        **kwargs,
    ) -> T:
        """Execute ``cls.read()``.

        This method generalises parallel reading of lists of files for
        any input class.


        This method also generalises downloading files from remote URLs
        over HTTP or via `fsspec`.
        """
        cls = self._cls

        # extract merge-specific kwargs by inspecting the merge method signature
        if merge_function is None:
            merge_function = self.merge
        merge_kwargs = self._merge_kwargs(kwargs, merge_function)

        # handle deprecated keyword
        if nproc := kwargs.pop("nproc", None):
            warnings.warn(
                f"the 'nproc' keyword to {cls.__name__}.read was renamed "
                "parallel; this warning will be an error in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
            parallel = nproc

        # set default cache based on environment
        if cache is None:
            cache = bool_env("GWPY_CACHE", default=False)

        # get the input as a list of inputs
        sources = self._format_input_list(source)

        # handle progress based on the number of inputs
        show_progress: str | bool = False  # single file download progress
        if len(sources) == 1:
            show_progress = verbose
            verbose = False
        if verbose is True and format:
            verbose = f"Reading ({format})"
        elif verbose is True:
            verbose = "Reading"
        if verbose:
            progress = progress_bar(
                total=len(sources),
                desc=verbose,
            )
        else:
            progress = None

        remote_kwargs = {
            key: kwargs.pop(key) for key in (
                "sources",
                "http_headers",
                "use_fsspec",
                "fsspec_kwargs",
            ) if key in kwargs
        }

        # single file reader
        def _read(arg: NamedReadable) -> T:
            ctx: AbstractContextManager[NamedReadable | BinaryIO]
            # if arg is a str, presume it represents a URI, so try and open it
            if isinstance(arg, str):
                ctx = open_remote_file(
                    arg,
                    cache=cache,
                    show_progress=show_progress,
                    **remote_kwargs,
                )
            # otherwise just pass it along unmodified
            else:
                ctx = nullcontext(arg)

            with ctx as file:
                return self.registry.read(
                    self._cls,
                    file,
                    *args,
                    format=format,
                    cache=cache,
                    **kwargs,
                )

        # read all files in parallel threads
        outputs: list[T] = []
        with ThreadPoolExecutor(
            max_workers=parallel,
        ) as pool:
            futures = [pool.submit(_read, source) for source in sources]
            for future in as_completed(futures):
                outputs.append(future.result())
                if progress:
                    progress.update(1)

        # Merge results
        return merge_function(outputs, **merge_kwargs)


class UnifiedWrite(astropy_registry.UnifiedReadWrite, Generic[T]):
    """Base ``Class.write()`` implementation."""

    def __init__(
        self,
        instance: T,
        cls: type[T],
        registry: UnifiedIORegistry = default_registry,
    ) -> None:
        """Initialise a new `UnifiedWrite` instance."""
        super().__init__(
            instance,
            cls,
            "write",
            registry=registry,
        )

    def __call__(
        self,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> None:
        """Execute ``instance.write()``."""
        instance = self._instance
        return self.registry.write(instance, *args, **kwargs)


# -- utilities -----------------------

def inherit_unified_io(klass: type) -> type:
    """Re-register all Unified I/O readers/writers/identifiers from a parent to a child.

    Only works with the first parent in the inheritance tree.

    This allows the Unified I/O registrations for the child class to be
    modified independently of the parent.
    """
    parent = klass.__mro__[1]
    parent_registry = parent.read.registry  # type: ignore[attr-defined]
    child_registry = klass.read.registry  # type: ignore[attr-defined]
    for row in parent_registry.get_formats(data_class=parent):
        name = row["Format"]

        # read
        if row["Read"].lower() == "yes":
            child_registry.register_reader(
                name,
                klass,
                parent_registry.get_reader(name, parent),
                force=False,
            )
        # write
        if row["Write"].lower() == "yes":
            child_registry.register_writer(
                name,
                klass,
                parent_registry.get_writer(name, parent),
                force=False,
            )
        # identify
        if row["Auto-identify"].lower() == "yes":
            child_registry.register_identifier(
                name,
                klass,
                parent_registry._identifiers[(name, parent)],
                force=False,
            )
    return klass
