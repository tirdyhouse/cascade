#!/usr/bin/env bash
# Verify that the target filesystem performs native GPUDirect Storage I/O.
set -euo pipefail

gds_mount="${1:-/mnt/nvfile}"
gds_tools="${GDS_TOOLS:-/usr/local/cuda-13.2/gds/tools}"
gds_cfg=$(mktemp /tmp/cufile-direct-only.XXXXXX.json)
gds_log=$(mktemp /tmp/cufile-direct-only.XXXXXX.log)
gds_test_dir=$(mktemp -d "${gds_mount}/.gds-direct-only.XXXXXX")
gds_test_file="${gds_test_dir}/gdsio-256m.bin"

cleanup() {
  rm -f "$gds_test_file" "$gds_cfg" "$gds_log"
  rmdir "$gds_test_dir" 2>/dev/null || true
}
trap cleanup EXIT

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script as root." >&2
  exit 2
fi

test -x "$gds_tools/gdscheck"
test -x "$gds_tools/gdsio"
test -r /etc/cufile.json

sed -E \
  's/"allow_compat_mode"[[:space:]]*:[[:space:]]*true/"allow_compat_mode": false/' \
  /etc/cufile.json > "$gds_cfg"

if ! grep -Eq '"allow_compat_mode"[[:space:]]*:[[:space:]]*false' "$gds_cfg"; then
  echo "FAIL: unable to disable cuFile compatibility fallback." >&2
  exit 2
fi

echo "=== Mount ==="
findmnt -T "$gds_mount" -o TARGET,SOURCE,FSTYPE,OPTIONS

echo "=== NVFile build mode ==="
dmesg | grep -E 'nvfile: mount.*Built (with|without) NVFS RDMA support' | tail -n 1 || true

echo "=== nvidia-fs registrations ==="
cat /proc/driver/nvidia-fs/modules || true

echo "=== Strict GDS platform check ==="
CUFILE_ENV_PATH_JSON="$gds_cfg" "$gds_tools/gdscheck" -p | \
  grep -E 'BeeGFS|use_compat_mode|nvidia_fs version|GPU index' || true

echo "=== Strict native-GDS I/O test ==="
if CUFILE_ENV_PATH_JSON="$gds_cfg" \
   CUFILE_LOGGING_LEVEL=DEBUG \
   CUFILE_LOGFILE_PATH="$gds_log" \
   "$gds_tools/gdsio" \
     -f "$gds_test_file" -d 0 -m 0 -w 1 \
     -s 256M -i 1M -x 0 -I 1 -V
then
  echo "PASS: native GDS is active on ${gds_mount}."
else
  rc=$?
  echo "FAIL: native GDS is unavailable; compatibility fallback was disabled." >&2
  grep -Ei 'unsupported|compat|BeeGFS|NVFS|register|error' "$gds_log" || true
  exit "$rc"
fi
