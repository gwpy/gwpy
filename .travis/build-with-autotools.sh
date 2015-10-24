#!/bin/bash
#
# build a package with autotools

set -e

tarball=$1
shift
if [[ "`uname`" == "Darwin"* ]]; then
    target=`greadlink -f $1`
else
    target=`readlink -f $1`
fi
shift
configargs="$@"

# set paths
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${target}/lib/pkgconfig
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${target}/lib
for libd in "lib lib64"; do
    sited="${target}/${libd}/python${TRAVIS_PYTHON_VERSION}/site-packages"
    export PYTHONPATH=${PYTHONPATH}:${sited}
done

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
