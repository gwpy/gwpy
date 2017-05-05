#!/bin/bash
#
# Set environment for src builds for the GWpy travis-ci runner

# -- utilities

pip install --quiet bs4 six
find_latest_version() {
    python .travis/find-latest-release.py $@
}

LSCSOFT_URL="http://software.ligo.org/lscsoft/source"
LALSUITE_URL="${LSCSOFT_URL}/lalsuite"

# -- non-pure-python packages

echo "Querying for latest versions of LSCSoft packages:"

read NDS2_CLIENT_VERSION NDS2_CLIENT < <(python .travis/find-latest-release.py ${LSCSOFT_URL} nds2-client)
echo "   nds2-client:  ${NDS2_CLIENT_VERSION}"

read LDAS_TOOLS_AL_VERSION LDAS_TOOLS_AL < <(python .travis/find-latest-release.py ${LSCSOFT_URL} ldas-tools-al)
echo "   ldas-tools-l: ${LDAS_TOOLS_AL_VERSION}"

read FRAMECPP_VERSION FRAMECPP < <(python .travis/find-latest-release.py ${LSCSOFT_URL} ldas-tools-framecpp)
echo "   frameCPP:     ${FRAMECPP_VERSION}"

read LAL_VERSION LAL < <(python .travis/find-latest-release.py ${LALSUITE_URL} lal)
echo "   lal:          ${LAL_VERSION}"

read LALFRAME_VERSION LALFRAME < <(python .travis/find-latest-release.py ${LALSUITE_URL} lalframe)
echo "   lalframe:     ${LALFRAME_VERSION}"
