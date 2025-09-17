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

from collections.abc import Iterator
from typing import (
    Any,
    overload,
)

class availability:  # noqa: N801
    data: segment_list_type

class availability_list_type(list[availability]): ...  # noqa: N801

class buffer:  # noqa: N801
    channel: channel
    channel_type: int
    data: Any  # numpy array-like
    data_type: int
    gps_seconds: int
    gps_nanoseconds: int
    length: int
    name: str
    sample_rate: float
    signal_gain: float
    signal_slope: float
    signal_offset: float
    signal_units: str

    def release(self) -> None: ...

class channel:  # noqa: N801
    # Class constants
    MIN_SAMPLE_RATE: float
    MAX_SAMPLE_RATE: float
    DEFAULT_CHANNEL_MASK: int
    DEFAULT_DATA_MASK: int

    # Instance attributes accessed in GWpy
    name: str
    sample_rate: float
    signal_units: str | None
    data_type: int
    channel_type: int

    # Static methods
    @staticmethod
    def channel_type_to_string(channel_type: int) -> str: ...

    @staticmethod
    def data_type_to_string(data_type: int) -> str: ...

class connection:  # noqa: N801
    def __init__(
        self,
        hostname: str,
        port: int = ...,
    ) -> None: ...

    def open(self, start_time: int, end_time: int) -> None: ...

    def close(self) -> None: ...

    def count_channels(
        self,
        channel_glob: str = ...,
        channel_type_mask: int = ...,
        data_type_mask: int = ...,
        min_sample_rate: float = ...,
        max_sample_rate: float = ...,
    ) -> int: ...

    def current_epoch(self) -> segment: ...

    def fetch(
        self,
        gps_start: int,
        gps_stop: int,
        channel_names: list[str],
    ) -> list[buffer]: ...

    def find_channels(
        self,
        channel_glob: str = ...,
        channel_type_mask: int = ...,
        data_type_mask: int = ...,
        min_sample_rate: float = ...,
        max_sample_rate: float = ...,
    ) -> list[channel]: ...

    def get_availability(
        self,
        channels: list[str],
    ) -> availability_list_type:
        ...

    def get_host(self) -> str: ...

    def get_port(self) -> int: ...

    def get_parameter(self, parameter: str) -> str: ...

    def get_protocol(self) -> int: ...

    @overload
    def iterate(
        self,
        channel_names: list[str],
    ) -> Iterator[Iterator[buffer]]: ...
    @overload
    def iterate(
        self,
        stride: int,
        channel_names: list[str],
    ) -> Iterator[Iterator[buffer]]: ...
    @overload
    def iterate(
        self,
        gps_start: int,
        gps_stop: int,
        channel_names: list[str],
    ) -> Iterator[Iterator[buffer]]: ...
    @overload
    def iterate(
        self,
        gps_start: int,
        gps_stop: int,
        stride: int,
        channel_names: list[str],
    ) -> Iterator[Iterator[buffer]]: ...

    @overload
    def set_epoch(self, start_time: str) -> None: ...
    @overload
    def set_epoch(self, start_time: int, end_time: int) -> None: ...

    def set_parameter(self, parameter: str, value: str) -> bool: ...

class segment:  # noqa: N801
    gps_start: int
    gps_stop: int
    frame_type: str

    @overload
    def __init__(self, start: int, stop: int) -> None: ...
    @overload
    def __init__(self, frame_type: str, start: int, stop: int) -> None: ...

class segment_list_type(list[segment]): ...  # noqa: N801

class simple_segment:  # noqa: N801
    gps_start: int
    gps_stop: int
