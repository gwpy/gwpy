#!/bin/bash
#
# Set environment for src builds for the GWpy travis-ci runner

if [[ "${PRE}" == "--pre" ]]; then  # open conditional

# -- utilities

pip install --quiet bs4
find_latest_version() {
    python .travis/find-latest-release.py -o version $@
}

get_requirements_version() {
    local package=$1
    tarball=`grep $package requirements.txt`
    python -c "
import os.path
print(os.path.basename('$tarball').rsplit('-')[1][:-7])"
}

sedx() {
    if [ `uname -s` = "Darwin" ]; then
        sed -i '' $@
    else
        sed -i $@
    fi
}

LSCSOFT_URL="http://software.ligo.org/lscsoft/source"
LALSUITE_URL="${LSCSOFT_URL}/lalsuite"

# -- non-pure-python packages

LIBFRAME_VERSION=`find_latest_version ${LSCSOFT_URL} libframe`
LIBFRAME="${LSCSOFT_URL}/libframe-${LIBFRAME_VERSION}.tar.gz"

NDS2_CLIENT_VERSION=`find_latest_version ${LSCSOFT_URL} nds2-client`
NDS2_CLIENT="${LSCSOFT_URL}/nds2-client-${NDS2_CLIENT_VERSION}.tar.gz"

LDAS_TOOLS_VERSION=`find_latest_version ${LSCSOFT_URL} ldas-tools`
LDAS_TOOLS="${LSCSOFT_URL}/ldas-tools-${LDAS_TOOLS_VERSION}.tar.gz"

LAL_VERSION=`find_latest_version ${LALSUITE_URL} lal`
LAL="${LALSUITE_URL}/lal-${LAL_VERSION}.tar.gz"

LALFRAME_VERSION=`find_latest_version ${LALSUITE_URL} lalframe`
LALFRAME="${LALSUITE_URL}/lalframe-${LALFRAME_VERSION}.tar.gz"

# -- python packages

OLD_GLUE_VERSION=`get_requirements_version glue`
GLUE="${LSCSOFT_URL}/glue-${GLUE_VERSION}.tar.gz"
GLUE_VERSION=`find_latest_version ${LSCSOFT_URL} glue`
sedx 's/glue-'${OLD_GLUE_VERSION}'/glue-'${GLUE_VERSION}'/g' requirements.txt

OLD_DQSEGDB_VERSION=`get_requirements_version dqsegdb`
DQSEGDB="${LSCSOFT_URL}/dqsegdb-${DQSEGDB_VERSION}.tar.gz"
DQSEGDB_VERSION=`find_latest_version ${LSCSOFT_URL} dqsegdb`
sedx 's/dqsegdb-'${OLD_DQSEGDB_VERSION}'/dqsegdb-'${DQSEGDB_VERSION}'/g' requirements.txt

# -- clean up

fi  # close conditional

