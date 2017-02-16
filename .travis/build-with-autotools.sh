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

# check for existing file
if [ -f $builddir/.travis-src-file ] && [ `cat $builddir/.travis-src-file` == "$tarball" ]; then
    echo "Cached build directory found, not downloading tarball"
else
    echo "New build requested, downloading tarball..."
    rm -rf $builddir/
    mkdir -p $builddir
    wget $tarball -O `basename $tarball`
    tar -xf `basename $tarball` -C $builddir --strip-components=1
    echo $tarball > $builddir/.travis-src-file
fi

# always install and return
cd $builddir
if [ -f ./00boot ]; then
    ./00boot
elif [ -f ./autogen.sh ]; then
    ./autogen.sh
fi
./configure --enable-silent-rules --prefix=$target $@
make -j 2 --silent || make --silent
make install --silent
cd -
echo "----------------------------------------------------------------------"
echo "Successfully installed `basename ${tarball}`"
echo "----------------------------------------------------------------------"
