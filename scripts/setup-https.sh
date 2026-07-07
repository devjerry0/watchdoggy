#!/usr/bin/env bash
# Give the Pi's dashboard HTTPS with a household private CA.
# Browsers only allow the microphone (push-to-talk) and notifications on
# secure pages. This creates a "watchdoggy home CA" on the Pi, issues the
# dashboard a certificate signed by it, and serves the CA at /ca.pem so
# each of your devices can trust it once. After that: a normal padlock,
# no warnings, no renewals, no internet needed.
# Usage: ./scripts/setup-https.sh <user@host> [appdir]
set -euo pipefail
TARGET="${1:?usage: setup-https.sh <user@host> [appdir]}"
APPDIR="${2:-doggy}"
ssh "$TARGET" "APPDIR='$APPDIR' bash -s" <<'REMOTE'
set -euo pipefail
cd "$HOME/$APPDIR"
mkdir -p certs && chmod 700 certs
HOST="$(hostname).local"
IP="$(hostname -I | awk '{print $1}')"

if [ ! -f certs/ca.pem ]; then
  echo "==> creating the home CA (once)"
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
    -keyout certs/ca-key.pem -out certs/ca.pem -days 3650 -nodes \
    -subj "/CN=watchdoggy home CA/O=watchdoggy" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"
  chmod 600 certs/ca-key.pem
fi

echo "==> issuing the dashboard certificate (825 days, re-run to renew)"
openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout certs/key.pem -out certs/req.csr -nodes -subj "/CN=$HOST"
openssl x509 -req -in certs/req.csr -CA certs/ca.pem -CAkey certs/ca-key.pem \
  -CAcreateserial -out certs/cert.pem -days 825 \
  -extfile <(printf "subjectAltName=DNS:%s,IP:%s\nextendedKeyUsage=serverAuth\nbasicConstraints=CA:FALSE\n" "$HOST" "$IP")
rm -f certs/req.csr
chmod 600 certs/key.pem

grep -q '^DOGGY_SSL_CERT=' .env || cat >> .env <<EOF
DOGGY_SSL_CERT=certs/cert.pem
DOGGY_SSL_KEY=certs/key.pem
DOGGY_CA_CERT=certs/ca.pem
EOF
sudo systemctl restart doggy
echo "==> dashboard now at https://$HOST:8000"
echo "    On each device, download https://$HOST:8000/ca.pem (one warning"
echo "    this first time) and trust it:"
echo "      iPhone/iPad: open the file, install the profile, then Settings >"
echo "        General > About > Certificate Trust Settings > enable it"
echo "      Mac: double-click it in Keychain Access, set Trust to Always"
echo "      Android: Settings > Security > Install a certificate > CA"
echo "    After that the padlock is normal everywhere. No renewals needed"
echo "    until $(date -d '+825 days' '+%Y-%m' 2>/dev/null || echo '~2028'); re-run this script then."
REMOTE
