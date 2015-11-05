#!/bin/bash
#
# build a package with autotools

set -e

tarball=$1
shift
configargs="$@"

echo "----------------------------------------------------------------------"
echo "Installing from ${tarball}"

# set install target to python prefix
target=`python -c "import sys; print(sys.prefix)"`
echo "Will install into ${target}"

# move to build directory
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
