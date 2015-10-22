#!/bin/bash
#
# build a package with autotools

set -ev

tarball=$1
shift
if [[ "`uname`" == "Darwin"* ]]; then
    target=`greadlink -f $1`
else
    target=`readlink -f $1`
fi
shift
configargs="$@"

if [ -d ${target}/lib/pkgconfig ] && [ "$(ls -A ${target}/lib/pkconfig)" ]; then
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
./configure --prefix=$target $@
make #-j
make install

export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${target}/lib/pkgconfig
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${target}/lib
for libd in "lib lib64"; do
    sited="${target}/${libd}/python${TRAVIS_PYTHON_VERSION}/site-packages"
    [ -d ${sited} ] && export PYTHONPATH=${PYTHONPATH}:${sited}
done
cd -
rm -rf ${builddir}
