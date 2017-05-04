#!/bin/bash

# build and install numpy first
pip install ${PIP_FLAGS} "numpy>=1.10"

# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

# use g++ 4.7 (mainly for framecpp)
export CXX=`which g++-4.7`
echo "Set CXX=${CXX}"

# get latest version of packages
. .travis/find-latest-releases.sh

FAILURES=""

# -- LAL ----------------------------------------------------------------------

bash -e .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-lal ${LAL} \
    --quiet --enable-silent-rules \
    --enable-swig-python --disable-swig-octave --disable-doxygen \
    --with-hdf5=no || FAILURES="$FAILURES lal"

# -- framecpp -----------------------------------------------------------------

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-ldas-tools-al ${LDAS_TOOLS_AL} \
    --quiet --enable-silent-rules \
    --enable-python || FAILURES="$FAILURES ldas-tools-al"

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-framecpp ${FRAMECPP} \
    --quiet --enable-silent-rules \
    --enable-python --disable-latex \
    --enable-fast-install --with-optimization=none \
    || FAILURES="$FAILURES framecpp"

# -- lalframe -----------------------------------------------------------------

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-lalframe ${LALFRAME} \
    --quiet --enable-silent-rules \
    --enable-swig-python --disable-swig-octave --disable-doxygen \
    || FAILURES="$FAILURES lalframe"

# -- NDS2 ---------------------------------------------------------------------

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-nds2-client ${NDS2_CLIENT} \
    -DENABLE_SWIG_JAVA=false -DENABLE_SWIG_MATLAB=false \
    || FAILURES="$FAILURES nds2-client"

# -- exit ---------------------------------------------------------------------

if [[ -n "${FAILURES+x}" ]]; then
    echo "The following builds failed: ${FAILURES}"
fi
