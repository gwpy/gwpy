#!/bin/bash

# enable strict test flags
if [ "$STRICT" = true ]; then
    _strict="-x --strict"
else
    _strict=""
fi

coverage run --source=gwpy --omit="gwpy/tests/*,gwpy/*version*,gwpy/utils/sphinx/*" -m py.test -n 2 -v -r s ${_strict} gwpy/
