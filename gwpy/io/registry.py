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
import logging
import os
import pydoc
import re
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
    from collections.abc import (
        Callable,
        Iterable,
        Sequence,
    )
    from contextlib import AbstractContextManager
    from typing import (
        Any,
        BinaryIO,
        Literal,
        Never,
    )

    from .utils import (
        FileLike,
        FileSystemPath,
        NamedReadable,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "GetExceptionGroup",
    "UnifiedFetch",
    "UnifiedFetchRegistry",
    "UnifiedGet",
    "UnifiedGetRegistry",
    "UnifiedIORegistry",
    "UnifiedRead",
    "UnifiedWrite",
    "default_registry",
    "identify_factory",
    "inherit_unified_io",
]

# Type variable for Generic classes
T = TypeVar("T")


class GetExceptionGroup(ExceptionGroup):
    """Exception group raised by ``Klass.get()`` when all sources fail.

    This is a subclass of `ExceptionGroup` that can be used to identify
    exceptions raised specifically by the ``Klass.get()`` class when all
    sources fail to get data.
    """


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


def _list_identifier(
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
            _list_identifier(identifier),
            force=force,
        )

    register_identifier.__doc__ = (
        astropy_registry.UnifiedIORegistry.register_identifier.__doc__
    )


#: The default unified I/O registry instance.
#:
#: Most classes should use this registry unless they have a specific reason
#: to use a custom registry.
default_registry = UnifiedIORegistry()



class MergeProtocol(Protocol[T]):
    """Typing protocol for merge functions used in `UnifiedRead`."""

    def __call__(self, items: Sequence[T]) -> T:
        """Merge a sequence of items into a single item."""


class UnifiedRead(astropy_registry.UnifiedReadWrite, ABC, Generic[T]):
    """Base ``Class.read()`` implementation that handles parallel reads.

    Each parallel read must return an instance of the same type
    as the input class, and the results are merged by the provided
    ``merge_function`` which should have the following signature::

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
                    show_progress=bool(show_progress),
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

        outputs: list[T] = []

        # read files in series
        if parallel <= 1:
            for src in sources:
                outputs.append(_read(src))
                if progress:
                    progress.update(1)
        # read all files in parallel threads
        else:
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


# -- Fetch ---------------------------
# custom registry to support Klass.fetch

class UnifiedFetchRegistry(astropy_registry.UnifiedInputRegistry):
    """`UnifiedInputRegistry` hacked to support a ``.fetch()`` method.

    Fetch is a read-like operation that does not take a file or file-like
    object as input, but instead takes other arguments to fetch data from
    a single source, e.g. a database or web API.
    """

    name: Literal["read", "fetch", "get"] = "fetch"

    def __init__(self) -> None:
        """Initialise a new `UnifiedFetchRegistry` instance."""
        super().__init__()
        self._registries[self.name] = self._registries.pop("read")
        self._registries_order = (self.name, "identify")

    def _update__doc__(self, data_class: type, readwrite: str) -> None:
        """Update the docstring for ``data_class.<readwrite>()`` method."""
        if readwrite == "read":
            readwrite = self.name
        super()._update__doc__(data_class, readwrite)

        # replace the word 'format' with 'source' in the docstring
        class_readwrite_func = getattr(data_class, readwrite)
        doc = class_readwrite_func.__doc__
        if doc is None:
            return
        for search, replace in [
            # Leader for the sources table
            ("built-in formats", "built-in sources"),
            # Column heading in the sources table
            (r"\n(\s*)?Format", r"\n\1Source"),
        ]:
            doc = re.sub(search, replace, doc)
        class_readwrite_func.__class__.__doc__ = doc

    def _get_valid_format(
        self,
        mode: Literal["read", "fetch", "get"],
        cls: type,
        path: FileSystemPath | None,
        fileobj: FileLike | None,
        args: tuple,
        kwargs: dict,
    ) -> str:
        """Return the first valid format that can be used."""
        if mode.lower() == "read":
            mode = self.name
        return super()._get_valid_format(mode, cls, path, fileobj, args, kwargs)


fetch_registry = UnifiedFetchRegistry()


class UnifiedFetch(UnifiedRead[T], Generic[T]):
    """Base ``Class.fetch()`` implementation."""

    method: Literal["fetch", "get"] = "fetch"

    def merge(self, *args, **kwargs) -> Never:
        """Fetch does not support merging."""
        msg = f"{self._cls.__name__}.{self.method} does not support merging"
        raise NotImplementedError(msg)

    def __init__(
        self,
        instance: T,
        cls: type[T],
        registry: UnifiedFetchRegistry = fetch_registry,
    ) -> None:
        """Initialise a new `UnifiedFetch` instance."""
        super().__init__(
            instance,
            cls,
            registry=registry,
        )

    def __call__(  # type: ignore[override]
        self,
        *args,  # noqa: ANN002
        source: str | None = None,
        **kwargs,
    ) -> T:
        """Execute ``cls.fetch()``."""
        # 'read' using the registered format.
        return self.registry.read(
            self._cls,
            *args,
            format=source,
            **kwargs,
        )

    def help(
        self,
        source: str | None = None,
        out: FileLike | None = None,
    ) -> None:
        """Output help documentation for the specified `UnifiedFetch` ``source``.

        By default the help output is printed to the console via ``pydoc.pager``.
        Instead one can supplied a file handle object as ``out`` and the output
        will be written to that handle.

        Parameters
        ----------
        source : str
            `UnifiedFetch` source (format) name, e.g. 'sql' or 'gwosc'.

        out : None or file-like
            Output destination (default is stdout via a pager)
        """
        cls = self._cls

        # Get reader or writer function associated with the registry
        get_func = self._registry.get_reader
        try:
            if source:
                read_write_func = get_func(source, cls)
        except astropy_registry.IORegistryError as err:
            reader_doc = "ERROR: " + str(err)
        else:
            if source:
                # Format-specific
                header = (
                    f"{cls.__name__}.{self.method}(source='{source}') documentation\n"
                )
                doc = read_write_func.__doc__
            else:
                # General docs
                header = f"{cls.__name__}.{self.method} general documentation\n"
                doc = getattr(cls, self.method).__doc__

            reader_doc = re.sub(".", "=", header)
            reader_doc += header
            reader_doc += re.sub(".", "=", header)
            reader_doc += os.linesep
            if doc is not None:
                reader_doc += inspect.cleandoc(doc)

        if out is None:
            pydoc.pager(reader_doc)
        else:
            out.write(reader_doc)


# -- Get -----------------------------
# custom registry to support Klass.get

class UnifiedGetRegistry(UnifiedFetchRegistry):
    """Unified I/O registry for providing a multi-source ``.get()`` method.

    Each registered reader should work with the target class as normal.

    Each registered identified should return an iterable (generator) of
    keyword argument sets that should be tried, in order, to provide the
    necessary data.

    See `gwpy.timeseries.io.nds2` for a working example.
    """

    name = "get"
    _identifiers: dict[tuple[str, type], Callable[..., Sequence[dict[str, Any]]]]

    def identify_sources(
        self,
        source: str | Iterable[str | dict[str, Any]] | None,
        data_class_required: type,
        args: list[Any],
        kwargs: dict,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Identify all valid sources for the given arguments.

        Unlike the standard registry `identify_format` method, this returns
        a `list` of `dict`, each of which should be tried in turn with the
        keyword arguments applied for each source.

        This allows the `Klass.get` method to try multiple different sources
        to get some data.
        """
        # format input
        if isinstance(source, str):
            source = [source]
        sources: list[dict[str, Any]] = []
        source_names: set[str] = set()
        for src in source or []:
            if isinstance(src, str):
                src = {"source": src}  # noqa: PLW2901
            if not isinstance(src, dict):
                msg = "source must be a string or a dictionary of keyword arguments"
                raise TypeError(msg)
            try:
                source_names.add(src["source"])
            except KeyError as exc:
                msg = "source dictionary must contain a 'source' key"
                raise ValueError(msg) from exc
            sources.append(src)

        out: list[tuple[str, dict[str, Any]]] = []

        def _match_source(
            source: str | None = None,
            **source_kw,
        ) -> Iterable[tuple[str, Any]]:
            """Return all matching sources for the given source name and keywords.

            Each source identifier function returns a list of keyword argument dicts
            that should be tried in order.
            If the ``priority`` key is present in the dict, the results are
            sorted by priority across all registered sources
            (lowest first, default is 10).
            """
            match: list[tuple[str, dict[str, Any]]] = []
            for data_source, data_class in self._identifiers:
                # If the user specified a source, and it wasn't this one, skip it
                if source is not None and data_source != source:
                    continue
                # Exclude identifiers for a parent class where the child class
                # has its own registration _for this source type_.
                if not self._is_best_match(
                    data_class_required,
                    data_class,
                    filter(lambda klass: klass[0] == data_source, self._identifiers),
                ):
                    continue
                if new := self._identifiers[(data_source, data_class)](
                    self.name,
                    *args,
                    **kwargs,
                    **source_kw,
                ):
                    match.extend((data_source, kw) for kw in new)
            match.sort(key=lambda x: x[1].pop("priority", 10))  # sort by priority
            return match

        if not sources:
            sources = [{"source": None}]

        for source in sources:
            out.extend(_match_source(**source))

        return out


get_registry = UnifiedGetRegistry()


class UnifiedGet(UnifiedFetch, Generic[T]):
    """Unified I/O ``.get()`` implementation.

    This is similar to `UnifiedRead` or `UnifiedFetch` except that the getter
    will iterate over multiple sources returned by the
    `UnifiedGetRegistry.identify_sources` method, to retrieve the data any
    way it can.
    """

    method = "get"

    def __init__(
        self,
        instance: T,
        cls: type[T],
        registry: UnifiedGetRegistry = get_registry,
        module: str | None = None,
    ) -> None:
        """Initialise a new `UnifiedFetch` instance."""
        super().__init__(
            instance,
            cls,
            registry=registry,
        )
        self.logger = logging.getLogger(module or cls.__module__)

    def __call__(  # type: ignore[override]
        self,
        *args,
        source: str | list[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> T:
        """Execute ``cls.get()``."""
        reg = self.registry
        sources = reg.identify_sources(
            source,
            self._cls,
            args,
            kwargs,
        )
        nsources = len(sources)
        if not nsources:
            msg = "no valid sources found"
            raise ValueError(msg)
        self.logger.info("Found %d possible sources", nsources)

        errors = []

        # loop over each source
        for i, (src, source_kw) in enumerate(sources, start=1):
            self.logger.info("Attemping access with '%s' [%d/%d]", src, i, nsources)
            self.logger.debug(
                "Using options: %s",
                {k: v for k, v in source_kw.items() if v is not None},
            )
            try:
                return self._get(src, source_kw, *args, **kwargs)
            except Exception as exc:
                exc.add_note(f"Error getting data from {src} [{i}/{nsources}]")
                errors.append(exc)
                self.logger.debug(
                    "Failed to get data with %s: %s: %s",
                    src,
                    type(exc).__name__,
                    str(exc),
                )

        msg = "failed to get data from any source"
        raise GetExceptionGroup(msg, errors)

    def _get(
        self,
        source: str,
        source_kw: dict[str, Any],
        *args,
        **kwargs,
    ) -> T:
        """Try and get data from a single source."""
        getter = self.registry.get_reader(source, self._cls)

        # parse arguments
        sig = inspect.signature(getter)

        # Check if getter accepts **kwargs (VAR_KEYWORD parameter)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_keyword:
            # Pass all kwargs except framework parameters handled by __call__
            framework_params = {"source"}
            source_kw |= {
                key: val for key, val in kwargs.items()
                if key not in framework_params and val is not None
            }
        else:
            # Only pass kwargs that match explicitly named parameters
            params = [
                p.name
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            source_kw |= {
                key: val for key, val in kwargs.items()
                if key in params and val is not None
            }

        return getter(
            *args,
            **source_kw,
        )


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
