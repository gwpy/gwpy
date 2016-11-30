#!/bin/bash

# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

# build a newer version of swig
bash -e .travis/build-with-autotools.sh swig-${SWIG_VERSION}/${TRAVIS_PYTHON_VERSION} ${SWIG_}

# build FFTW3 (double and float)
bash -e .travis/build-with-autotools.sh fftw-${FFTW_VERSION}/${TRAVIS_PYTHON_VERSION} ${FFTW} --enable-shared=yes
bash -e .travis/build-with-autotools.sh fftw-${FFTW_VERSION}-float/${TRAVIS_PYTHON_VERSION} ${FFTW} --enable-shared=yes --enable-float


# build LAL packages
bash -e .travis/build-with-autotools.sh lal-${LAL_VERSION}/${TRAVIS_PYTHON_VERSION} ${LAL} --enable-swig-python
