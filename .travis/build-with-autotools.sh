#!/bin/bash
#
# build a package with autotools

set -e

tarball=$1
target=$2

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
./configure --prefix=`readlink -f $target` --enable-silent-rules --quiet
make
make install

export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${target}/lib/pkgconfig
