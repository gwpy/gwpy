#!/bin/bash
#
# build a package with autotools

set -e

tarball=$1
shift
target=`readlink -f $1`
shift
configargs="$@"

if [ "$(ls -A ${target}/lib/pkconfig)" ]; then
    echo "Target pkg-config directory is not empty, presuming successful cached build, will not build this package"
    return 0
fi

builddir="build_$RANDOM"
mkdir -p $builddir
echo "Building into $builddir"
# untar
wget $tarball --quiet -O `basename $tarball`
tar -zxf `basename $tarball` -C $builddir --strip-components=1
cd $builddir
if [ -f ./00boot ]; then
    ./00boot
fi
./configure --prefix=$target --enable-silent-rules --quiet $@
make -j
make install

export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${target}/lib/pkgconfig
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${target}/lib
if [ -d ${target}/lib/python${TRAVIS_PYTHON_VERSION}/site-packages ]; then
  export PYTHONPATH=${PYTHONPATH}:${target}/lib/python${TRAVIS_PYTHON_VERSION}/site-packages
fi
cd -
rm -rf ${builddir}
