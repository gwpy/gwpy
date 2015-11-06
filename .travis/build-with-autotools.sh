#!/bin/bash
#
# build a package with autotools

set -e

builddir=$1
tarball=$2
shift 2
configargs="$@"

echo "----------------------------------------------------------------------"
echo "Installing from ${tarball}"

# set install target to python prefix
target=`python -c "import sys; print(sys.prefix)"`
echo "Will install into ${target}"
echo "Building into $builddir"

# dont rebuild from scratch if not required
if [ -d $builddir ]; then
    echo "Cached build directory found, not downloading tarball"
else
    mkdir -p $builddir
    wget $tarball --quiet -O `basename $tarball`
    tar -zxf `basename $tarball` -C $builddir --strip-components=1
fi

# always install and return
cd $builddir
if [ -f ./00boot ]; then
    ./00boot
fi
./configure --enable-silent-rules --quiet --prefix=$target $@
make #-j
make install
cd -
