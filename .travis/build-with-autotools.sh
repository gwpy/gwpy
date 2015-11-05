#!/bin/bash
#
# build a package with autotools

set -e

tarball=$1
shift
configargs="$@"

# set install target to python prefix
target=`python -c "import sys; print(sys.prefix)"`

# set paths

# check for cached build
if [ -d ${target}/lib/pkgconfig ]; then
    if [ "$(ls -A ${target}/lib/pkconfig)" ]; then
        echo "Target pkg-config directory is not empty, presuming successful cached build, will not build this package"
        return 0
    fi
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
./configure --enable-silent-rules --quiet --prefix=$target $@
make #-j
make install

cd -
rm -rf ${builddir}
echo "Updated PYTHONPATH to"
echo ${PYTHONPATH}
