#!/bin/bash
#
# Install and configure CVMFS
#

set -e

if [ "$GITLAB_CI" = "true" ] && [[ "$CI_RUNNER_TAGS" != *"linux"* ]]; then
echo -e "\x1B[94mSkipping CVMFS setup on this platform\x1B[0m"
exit 0
fi

echo -e "\x1B[92mConfiguring /cvmfs/gwosc.osgstorage.org\x1B[0m"

# configure CVMFS Apt repo
export DEBIAN_FRONTEND=noninteractive
apt-get -yqq update
apt-get -yqq install curl lsb-release
curl -sLO https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb
dpkg -i cvmfs-release-latest_all.deb
rm -f cvmfs-release-latest_all.deb

# configure CVMFS-contrib Apt repo
curl -sLO https://ecsft.cern.ch/dist/cvmfs/cvmfs-contrib-release/cvmfs-contrib-release-latest_all.deb
dpkg -i cvmfs-contrib-release-latest_all.deb
rm -f cvmfs-contrib-release-latest_all.deb

# install cvmfs
apt-get -yqq update
apt-get -yq install \
  cvmfs \
  cvmfs-config-osg \
;

# configure CVMFS client
cvmfs_config setup
cat > /etc/cvmfs/default.local << EOF
CVMFS_REPOSITORIES=gwosc.osgstorage.org
CVMFS_HTTP_PROXY=DIRECT
EOF

cvmfs_config probe
