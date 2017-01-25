#!/bin/bash

# find version-specific requirements file if given
if [ -f ./requirements-${TRAVIS_PYTHON_VERSION}.txt ]; then
    _reqfile="./requirements-${TRAVIS_PYTHON_VERSION}.txt"
# or use the default one (python3)
else
    _reqfile="./requirements.txt"
fi

# execute the install
pip install ${PIP_FLAGS} -r ${_reqfile}
