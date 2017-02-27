#!/bin/bash

# enable strict test flags
if [ "$STRICT" = true ]; then
    _strict="-x --strict"
else
    _strict=""
fi

coverage run --source=gwpy --omit="gwpy/tests/*,gwpy/*version*,gwpy/utils/sphinx/*,gwpy/table/rate.py,gwpy/table/utils.py,gwpy/table/rec.py,gwpy/table/io/ascii.py" -m py.test -v -r s ${_strict} gwpy/
