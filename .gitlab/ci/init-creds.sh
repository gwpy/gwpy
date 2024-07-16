#!/bin/bash
#
# Initialise credentials for jobs
#

if [ "${KERBEROS_KEYTAB}" = "" ]; then
echo -e "\x1B[94mNo KERBEROS_KEYTAB configured, skipping credential init.\x1B[0m"
exit 0
fi

# install tooling
conda install -c conda-forge -q -y ciecplib krb5

# unpack keytab
export KRB5_KTNAME="${CI_PROJECT_DIR}/.keytab"
echo "${KERBEROS_KEYTAB}" | base64 -d > ${KRB5_KTNAME}
chmod 0600 "${KRB5_KTNAME}"

# find principal
export KRB5_PRINCIPAL="$(klist -k ${KRB5_KTNAME} | tail -n 1 | awk '{print $2}')"

# get Kerberos TGT
kinit "${KRB5_PRINCIPAL}" -k -t "${KRB5_KTNAME}"
klist

# get LIGO.ORG X.509
ecp-get-cert -i login.ligo.org --kerberos
ecp-cert-info
