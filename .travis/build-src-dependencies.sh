#!/bin/bash

# don't install if doing a 'minimal' build
[[ "$MINIMAL" = true ]] && return

# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

FAILURES=""

# build a newer version of swig
bash .travis/build-with-autotools.sh swig-${SWIG_VERSION}/${TRAVIS_PYTHON_VERSION} ${SWIG_} || FAILURES="$FAILURES swig"

# build FFTW3 (double and float)
bash .travis/build-with-autotools.sh fftw-${FFTW_VERSION}/${TRAVIS_PYTHON_VERSION} ${FFTW} --enable-shared=yes || FAILURES="$FAILURES fftw"
bash .travis/build-with-autotools.sh fftw-${FFTW_VERSION}-float/${TRAVIS_PYTHON_VERSION} ${FFTW} --enable-shared=yes --enable-float || FAILURES="$FAILURES fftw-float"

# build ldas-tools-al
bash .travis/build-with-autotools.sh ldas-tools-al-${LDAS_TOOLS_AL_VERSION}/${TRAVIS_PYTHON_VERSION} ${LDAS_TOOLS_AL} || FAILURES="$FAILURES ldas-tools-al"

# build frame libraries
bash .travis/build-with-autotools.sh framecpp-${FRAMECPP_VERSION}/${TRAVIS_PYTHON_VERSION} ${FRAMECPP} --enable-python || FAILURES="$FAILURES framecpp"

bash .travis/build-with-autotools.sh libframe-${LIBFRAME_VERSION}/${TRAVIS_PYTHON_VERSION} ${LIBFRAME} || FAILURES="$FAILURES libframe"



# build LAL packages
bash .travis/build-with-autotools.sh lal-${LAL_VERSION}/${TRAVIS_PYTHON_VERSION} ${LAL} --enable-swig-python || FAILURES="$FAILURES lal"

bash .travis/build-with-autotools.sh lalframe-${LALFRAME_VERSION}/${TRAVIS_PYTHON_VERSION} ${LALFRAME} --enable-swig-python || FAILURES="$FAILURES lalframe"

# build NDS2 client
bash .travis/build-with-autotools.sh nds2-client-${NDS2_CLIENT_VERSION}/${TRAVIS_PYTHON_VERSION} ${NDS2_CLIENT} --disable-swig-java --disable-mex-matlab || FAILURES="$FAILURES nds2-client"

if [[ -n "${FAILURES+x}" ]]; then
    echo "The following builds failed: ${FAILURES}"
fi
