# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2023)
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

"""Tests for :mod:`gwpy.utils.sphinx.zenodo`.
"""

from functools import wraps

import pytest

import requests

from ...testing.errors import pytest_skip_network_error
from ..sphinx import zenodo as gwpy_zenodo

GITHUB_RELEASE_API_URL = "https://api.github.com/repos/gwpy/gwpy/releases"

# simplified output of zenodo API query (as of 18/Oct/2023)
MOCK_ZENODO_API_JSON = {
    "hits": {
        "hits": [
            {
                "conceptdoi": "10.5281/zenodo.597016",
                "conceptrecid": "597016",
                "created": "2023-10-05T10:29:34.092078+00:00",
                "doi": "10.5281/zenodo.8409995",
                "doi_url": "https://doi.org/10.5281/zenodo.8409995",
                "files": None,
                "id": 8409995,
                "links": None,
                "metadata": {
                    "title": "gwpy/gwpy: GWpy 3.0.7",
                    "version": "v3.0.7",
                },
                "title": "gwpy/gwpy: GWpy 3.0.7",
            },
            {
                "conceptdoi": "10.5281/zenodo.597016",
                "conceptrecid": "597016",
                "created": "2023-10-05T09:51:21.482810+00:00",
                "doi": "10.5281/zenodo.8409892",
                "doi_url": "https://doi.org/10.5281/zenodo.8409892",
                "files": None,
                "links": None,
                "metadata": {
                    "title": "gwpy/gwpy: GWpy 3.0.6",
                    "version": "v3.0.6",
                },
                "title": "gwpy/gwpy: GWpy 3.0.6",
            },
        ],
    },
}
MOCK_ZENODO_API_RST = """
-----
3.0.7
-----

.. image:: https://zenodo.example.com/badge/doi/10.5281/zenodo.8409995.svg
    :alt: gwpy/gwpy: GWpy 3.0.7 Zenodo DOI badge
    :target: https://doi.org/10.5281/zenodo.8409995

-----
3.0.6
-----

.. image:: https://zenodo.example.com/badge/doi/10.5281/zenodo.8409892.svg
    :alt: gwpy/gwpy: GWpy 3.0.6 Zenodo DOI badge
    :target: https://doi.org/10.5281/zenodo.8409892
""".strip()


def pytest_skip_zenodo_http_errors(func):
    """Execute `func` but skip if it raises a known server-side error.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as exc:  # pragma: no cover
            if (
                # API rate limit
                str(exc).startswith("403 Client Error: rate limit exceeded")
                # Bad Gateway
                or exc.response.status_code == 502
            ):
                pytest.skip(str(exc))
            raise
    return wrapper


@pytest.fixture
@pytest_skip_network_error
@pytest_skip_zenodo_http_errors
def latest():
    """Get the latest release of GWpy from the GitHub API.
    """
    resp = requests.get(
        GITHUB_RELEASE_API_URL,
        params={"per_page": 1},
    )
    resp.raise_for_status()
    return resp.json()[0]


@pytest_skip_network_error
@pytest_skip_zenodo_http_errors
def test_zenodo_format_citations_latest(latest):
    """Check that :func:`gwpy.utils.sphinx.zenodo.format_citations` includes
    the latest actual release in the output.
    """
    rst = gwpy_zenodo.format_citations(
        597016,
    )
    latestversion = latest["tag_name"].lstrip("v")
    latesthead = "-" * len(latestversion)
    assert f"""
{latesthead}
{latestversion}
{latesthead}""".strip() in rst


def test_zenodo_format_citations_mock(requests_mock):
    """Check that :func:`gwpy.utils.sphinx.zenodo.format_citations` correctly
    formats the JSON response it gets from Zenodo.

    This uses a mocked API response based on the actual response as of
    18/Oct/2023.
    """
    # mock
    requests_mock.get(
        "https://zenodo.example.com/api/records",
        json=MOCK_ZENODO_API_JSON,
    )

    # run
    rst = gwpy_zenodo.format_citations(
        597016,
        url="https://zenodo.example.com",
    )

    # check
    assert rst.strip() == MOCK_ZENODO_API_RST.strip()


def test_zenodo_main(requests_mock, tmp_path):
    """Check that the command-line entry point for `gwpy.utils.sphinx.zenodo`
    works correctly (with a mocked API response).
    """
    # mock
    requests_mock.get(
        "https://zenodo.example.com/api/records",
        json=MOCK_ZENODO_API_JSON,
    )

    # run
    out = tmp_path / "tmp.txt"
    gwpy_zenodo.main([
        "597016",
        "--url", "https://zenodo.example.com",
        "--output-file", str(out),
    ])

    # check
    assert out.read_text().strip() == MOCK_ZENODO_API_RST.strip()
