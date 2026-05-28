#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN="$ROOT_DIR/bin/disk-cache"
CACHE_DIR_CREATED=0
META_DIR_CREATED=0
if [[ -z "${CACHE_DIR:-}" ]]; then
  CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cascade-cache.XXXXXX")
  CACHE_DIR_CREATED=1
fi
if [[ -z "${META_DIR:-}" ]]; then
  META_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cascade-meta.XXXXXX")
  META_DIR_CREATED=1
fi
LISTEN_ADDR=${LISTEN_ADDR:-127.0.0.1:19100}
BASE_URL="http://$LISTEN_ADDR"
PID=""

cleanup() {
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  if [[ ${KEEP_SMOKE_DIRS:-0} == 1 ]]; then
    echo "Keeping smoke directories: CACHE_DIR=$CACHE_DIR META_DIR=$META_DIR"
    return
  fi
  if [[ "$CACHE_DIR_CREATED" == 1 ]]; then
    rm -rf "$CACHE_DIR"
  fi
  if [[ "$META_DIR_CREATED" == 1 ]]; then
    rm -rf "$META_DIR"
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"
mkdir -p "$(dirname "$BIN")" "$CACHE_DIR" "$META_DIR"

if curl -fsS --max-time 1 "$BASE_URL/stats" >/dev/null 2>&1; then
  echo "refusing to run: $BASE_URL already responds; set LISTEN_ADDR to a free address" >&2
  exit 1
fi

echo "==> Building disk-cache"
(cd engine && go build -o "$BIN" ./cmd/disk-cache)

echo "==> Starting disk-cache on $LISTEN_ADDR"
"$BIN" -listen "$LISTEN_ADDR" -cache-path "$CACHE_DIR" -metadata-path "$META_DIR" -max-size 64MB >"$META_DIR/disk-cache.log" 2>&1 &
PID=$!

for _ in $(seq 1 50); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "disk-cache exited early" >&2
    cat "$META_DIR/disk-cache.log" >&2 || true
    exit 1
  fi
  if curl -fsS "$BASE_URL/stats" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "$BASE_URL/stats" >/dev/null

echo "==> Exercising put/get/exists/stats endpoints"
curl -fsS -X POST "$BASE_URL/put" \
  -H 'content-type: application/json' \
  -d '{"hash":4660,"file_path":"12/34/block.bin","size":128}' >/dev/null

exists=$(curl -fsS "$BASE_URL/exists?hash=1234")
[[ "$exists" == *'"exists":true'* ]] || {
  echo "unexpected /exists response: $exists" >&2
  exit 1
}

meta=$(curl -fsS "$BASE_URL/get?hash=1234")
[[ "$meta" == *'"Hash":4660'* ]] || {
  echo "unexpected /get response: $meta" >&2
  exit 1
}

stats=$(curl -fsS "$BASE_URL/stats")
[[ "$stats" == *'"BlocksStored":1'* ]] || {
  echo "unexpected /stats response: $stats" >&2
  exit 1
}

echo "==> Exercising record/match/chunk endpoints"
curl -fsS -X POST "$BASE_URL/record_batch" \
  -H 'content-type: application/json' \
  -d '{"token_ids":[11,22,33,44],"mm_hashes":["image-a"],"block_size":2}' >/dev/null

match=$(curl -fsS -X POST "$BASE_URL/match" \
  -H 'content-type: application/json' \
  -d '{"token_ids":[11,22,33,44],"mm_hashes":["image-a"],"block_size":2}')
[[ "$match" == *'"matched_tokens":4'* ]] || {
  echo "unexpected /match response: $match" >&2
  exit 1
}

prefix=$(python3 - <<'PY' "$match"
import json, sys
print(json.loads(sys.argv[1])["prompt_hash"])
PY
)

curl -fsS -X POST "$BASE_URL/chunk_put" \
  -H 'content-type: application/json' \
  -d "{\"prefix_key\":\"$prefix\",\"layer_name\":\"layer.0\",\"chunk_index\":0,\"num_tokens\":2}" >/dev/null

chunks=$(curl -fsS "$BASE_URL/chunk_list?prefix_key=$prefix&layer_name=layer.0")
[[ "$chunks" == *'"chunks":[0]'* ]] || {
  echo "unexpected /chunk_list response: $chunks" >&2
  exit 1
}

echo "Smoke test passed"
