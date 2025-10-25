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

"""Tests for :mod:`gwpy.io.scitokens`."""

import os
import time
from contextlib import nullcontext
from unittest import mock

import pytest

from .. import scitokens as io_scitokens

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# Rerun tests when the SciTokens demo token is invalid
rerun_invalid_token = pytest.mark.flaky(
    reruns=2,
    only_rerun=["InvalidTokenError"],
)

# Claims for the demo token
TEST_CLAIMS = {
    "aud": "https://gwpy.example.com",
    "scope": "gwpy.read gwpy.create gwpy.update",
}


@pytest.fixture
def tokenstr():
    """Return a serialized demo token using ``TEST_CLAIMS``."""
    demo = pytest.importorskip("scitokens.utils.demo")
    tok = demo.token(
        TEST_CLAIMS | {"exp": time.time() + 86400},
    )

    # ensure that iat is in the past
    time.sleep(2)

    return tok


@pytest.fixture
def token(tokenstr):
    """Return a demo `scitoken.SciToken` using ``TEST_CLAIMS``."""
    scitokens = pytest.importorskip("scitokens")
    return scitokens.SciToken.deserialize(tokenstr)


@pytest.fixture
def tokenpath(tmp_path):
    """Return a temporary path in which to write a token.

    The token is ***not*** written.
    """
    return tmp_path / "test.token"


@pytest.fixture
def htgettoken_main(tokenstr, tokenpath):
    """Mock `htgettoken.main` to write the demo ``tokenstr`` to ``tokenpath``."""
    # mock the whole htgettoken call
    def htgettoken(*args, **kwargs):  # noqa: ARG001
        with tokenpath.open("w") as file:
            print(tokenstr, file=file)

    with (
        mock.patch("htgettoken.main", side_effect=htgettoken) as mocker,
        mock.patch(
            "gwpy.io.scitokens.tempfile.TemporaryDirectory",
            return_value=nullcontext(tokenpath.parent),
        ),
    ):
        yield mocker


def assert_tokens_equal(result, expected):
    """Assert that token `scitokens.SciToken` objects have the same claims."""
    assert dict(result.claims()) == dict(expected.claims())


# -- find_token

@mock.patch.dict(os.environ)
@rerun_invalid_token
@pytest.mark.requires("igwn_auth_utils")
def test_find_token(token, tokenstr):
    """Test that `find_token` returns the right token."""
    os.environ["BEARER_TOKEN"] = tokenstr
    found = io_scitokens.find_token(
        TEST_CLAIMS["aud"],
        "gwpy.read",
        warn=True,
    )
    assert_tokens_equal(found, token)


@mock.patch.dict(os.environ)
@rerun_invalid_token
@pytest.mark.requires("igwn_auth_utils")
def test_find_token_error(tokenstr):
    """Test error handling in `find_token`."""
    os.environ["BEARER_TOKEN"] = tokenstr
    with (
        pytest.raises(
            RuntimeError,
            match="could not find a valid SciToken",
        ),
        pytest.warns(
            UserWarning,
            match="Validator rejected value of '.*' for claim 'scope'",
        ),
    ):
        io_scitokens.find_token(
            TEST_CLAIMS["aud"],
            "gwpy.delete",
            warn=True,
        )


# -- get_token

@rerun_invalid_token
@pytest.mark.requires("htgettoken", "scitokens", exc_type=(ImportError, OSError))
@pytest.mark.usefixtures("htgettoken_main")
def test_get_token(token, tokenpath):
    """Test `get_scitoken`."""
    # get a new token
    new = io_scitokens.get_scitoken(
        outfile=tokenpath,
        minsecs=600,
        quiet=False,
    )

    # check that we got the right token
    assert_tokens_equal(new, token)


@rerun_invalid_token
@pytest.mark.requires("htgettoken", "scitokens", exc_type=(ImportError, OSError))
def test_get_token_error_systemexit():
    """Test that `get_scitoken` handles `SystemExit` well."""
    with pytest.raises(
        RuntimeError,
        match="htgettoken failed",
    ) as exc_info:
        io_scitokens.get_scitoken(badkwarg=0)
    assert "no such option: --badkwarg" in str(exc_info.getrepr(chain=True))


# -- add_http_authorization_header

@rerun_invalid_token
@pytest.mark.requires("requests_scitokens")
def test_add_http_authorization_header_token(tokenstr, token):
    """Test `add_http_authorization_header`."""
    headers: dict[str, str] = {}
    io_scitokens.add_http_authorization_header(
        headers,
        token=token,
    )
    assert headers["Authorization"] == f"Bearer {tokenstr}"


@rerun_invalid_token
@mock.patch.dict("sys.modules", {"requests_scitokens": None})
def test_add_http_authorization_header_missing_import():
    """Test `add_http_authorization_header` handling of missing import."""
    with pytest.raises(ImportError):
        io_scitokens.add_http_authorization_header({})
    io_scitokens.add_http_authorization_header(headers := {}, error=False)
    assert "Authorization" not in headers


@mock.patch.dict(os.environ)
@pytest.mark.flaky(
    reruns=2,
    # find_token raises a RuntimeError on top of InvalidTokenError
    only_rerun=[RuntimeError],
)
@pytest.mark.requires("igwn_auth_utils", "requests_scitokens")
def test_add_http_authorization_header_find(tokenstr):
    """Test `add_http_authorization_header` calling out to find_token."""
    os.environ["BEARER_TOKEN"] = tokenstr
    headers: dict[str, str] = {}
    io_scitokens.add_http_authorization_header(
        headers,
        audience=TEST_CLAIMS["aud"],
        warn=True,
    )
    assert headers["Authorization"] == f"Bearer {tokenstr}"


@mock.patch.dict(os.environ)
@rerun_invalid_token
@pytest.mark.requires(
    "htgettoken",
    "igwn_auth_utils",
    "requests_scitokens",
    exc_type=(ImportError, OSError),
)
@pytest.mark.usefixtures("htgettoken_main")
def test_add_http_authorization_header_get():
    """Test `add_http_authorization_header` calling out to get_scitoken."""
    os.environ.pop("BEARER_TOKEN", None)
    os.environ.pop("BEARER_TOKEN_FILE", None)
    headers: dict[str, str] = {}
    io_scitokens.add_http_authorization_header(
        headers,
        audience=TEST_CLAIMS["aud"],
    )


@mock.patch("requests_scitokens.HTTPSciTokenAuth.find_token", side_effect=RuntimeError)
@mock.patch("gwpy.io.scitokens.get_scitoken", side_effect=RuntimeError)
@pytest.mark.requires("requests_scitokens")
def test_add_http_authorization_error(*_mocks):
    """Test `add_http_authorization_header` ``error`` keyword behaviour."""
    # check that the right error gets raised
    with pytest.raises(
        RuntimeError,
        match="failed to identify a valid SciToken",
    ):
        io_scitokens.add_http_authorization_header({})

    # check that error=False supresses the errors
    headers: dict[str, str] = {}
    io_scitokens.add_http_authorization_header(headers, error=False)
    assert "Authorization" not in headers
