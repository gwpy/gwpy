# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

# build a newer version of swig
source .travis/build-with-autotools.sh swig-${SWIG_VERSION} ${SWIG_}

# build FFTW3 (double and float)
source .travis/build-with-autotools.sh fftw-${FFTW_VERSION} ${FFTW} --enable-shared=yes
source .travis/build-with-autotools.sh fftw-${FFTW_VERSION}-float ${FFTW} --enable-shared=yes --enable-float

# build frame libraries
source .travis/build-with-autotools.sh ldas-tools-${LDAS_TOOLS_VERSION} ${LDAS_TOOLS}
source .travis/build-with-autotools.sh libframe-${LIBFRAME_VERSION} ${LIBFRAME}

# build LAL packages
source .travis/build-with-autotools.sh lal-${LAL_VERSION} ${LAL} --enable-swig-python
source .travis/build-with-autotools.sh lalframe-${LALFRAME_VERSION} ${LALFRAME} --enable-swig-python

# build NDS2 client
source .travis/build-with-autotools.sh nds2-client-${NDS2_CLIENT_VERSION} ${NDS2_CLIENT} --disable-swig-java --disable-mex-matlab
