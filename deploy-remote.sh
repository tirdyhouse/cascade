#!/usr/bin/env bash
# Build and stage Cascade control-plane binaries on a remote host.
#
# This script deliberately does not stop, restart, or delete any remote
# service. It is safe to run beside an existing LMCache/vLLM benchmark. Start
# only the staged Cascade components with the explicit commands in
# docs/BUILD_AND_DEPLOY.md after inspecting the target configuration.

set -euo pipefail

: "${REMOTE_HOST:?set REMOTE_HOST to the target hostname or IP}"
REMOTE_USER=${REMOTE_USER:-root}
REMOTE_DIR=${REMOTE_DIR:-/opt/cascade}
TARGET="${REMOTE_USER}@${REMOTE_HOST}"

echo "Staging Cascade control-plane source on ${TARGET}:${REMOTE_DIR}"
ssh "$TARGET" "mkdir -p '$REMOTE_DIR/bin' '$REMOTE_DIR/state'"

# Do not use --delete: the destination can contain operator-managed models,
# metadata, logs, or benchmark artifacts that this source checkout does not
# own.
rsync -az go.mod go.sum "$TARGET:$REMOTE_DIR/"
rsync -az engine/ "$TARGET:$REMOTE_DIR/engine/"
rsync -az adapter/ "$TARGET:$REMOTE_DIR/adapter/"

ssh "$TARGET" "
  set -e
  cd '$REMOTE_DIR'
  go build -o bin/cluster-server ./engine/cmd/cluster-server
  go build -o bin/c-agent ./engine/cmd/c-agent
  go build -o bin/disk-cache ./engine/cmd/disk-cache
  sha256sum bin/cluster-server bin/c-agent bin/disk-cache
"

echo "Staged binaries successfully. No remote process was stopped or started."
echo "Next: inspect docs/BUILD_AND_DEPLOY.md and start only the intended Cascade components with dedicated ports and state paths."
