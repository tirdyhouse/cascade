#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${MODEL_PATH:?MODEL_PATH is required, for example /path/to/Qwen2.5-7B-Instruct-AWQ}"
VLLM_BIN=${VLLM_BIN:-vllm}
PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONNECTOR_MODULE=${CONNECTOR_MODULE:-adapter.vllm.connector_v21}
DISK_CACHE_ADDR=${DISK_CACHE_ADDR:-127.0.0.1:19101}
VLLM_ADDR=${VLLM_ADDR:-127.0.0.1:18000}
CACHE_DIR_CREATED=0
META_DIR_CREATED=0
if [[ -z "${CACHE_DIR:-}" ]]; then
  CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/predict-vllm-cache.XXXXXX")
  CACHE_DIR_CREATED=1
fi
if [[ -z "${META_DIR:-}" ]]; then
  META_DIR=$(mktemp -d "${TMPDIR:-/tmp}/predict-vllm-meta.XXXXXX")
  META_DIR_CREATED=1
fi
DISK_CACHE_BIN=${DISK_CACHE_BIN:-$ROOT_DIR/bin/disk-cache}
DISK_CACHE_MAX_SIZE=${DISK_CACHE_MAX_SIZE:-20GB}
VLLM_LOG=${VLLM_LOG:-$META_DIR/vllm.log}
DISK_CACHE_LOG=${DISK_CACHE_LOG:-$META_DIR/disk-cache.log}
CHUNK_SIZE_MB=${CHUNK_SIZE_MB:-128}
STORAGE_BACKEND=${STORAGE_BACKEND:-posix}
TARGET_DEVICE=${TARGET_DEVICE:-cuda:0}
BLOCK_SIZE=${BLOCK_SIZE:-16}
REQUEST_REPETITIONS=${REQUEST_REPETITIONS:-350}
MAX_TOKENS=${MAX_TOKENS:-8}
TEMPERATURE=${TEMPERATURE:-0}
READY_TIMEOUT=${READY_TIMEOUT:-240}
SHUTDOWN_TIMEOUT=${SHUTDOWN_TIMEOUT:-5}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
DISABLE_PREFIX_CACHING=${DISABLE_PREFIX_CACHING:-1}
KEEP_VALIDATION_DIRS=${KEEP_VALIDATION_DIRS:-0}

# VLLM_EXTRA_ARGS is intentionally split on shell whitespace for simple flag
# tokens such as: --gpu-memory-utilization 0.90 --quantization awq. It is not a
# general quoting format; use env vars or extend this script if an argument value
# itself must contain whitespace or shell glob characters.

DC_PID=""
VLLM_PID=""
BASE_CACHE_URL="http://$DISK_CACHE_ADDR"
BASE_VLLM_URL="http://$VLLM_ADDR"

cleanup() {
  local exit_code=$?
  set +e

  if [[ "$exit_code" != 0 ]]; then
    echo "validation failed with exit code $exit_code" >&2
    echo "Logs: DISK_CACHE_LOG=$DISK_CACHE_LOG VLLM_LOG=$VLLM_LOG" >&2
    if [[ -f "$DISK_CACHE_LOG" ]]; then
      echo "--- disk-cache log tail ---" >&2
      tail -120 "$DISK_CACHE_LOG" >&2 || true
    fi
    if [[ -f "$VLLM_LOG" ]]; then
      echo "--- vLLM log tail ---" >&2
      tail -200 "$VLLM_LOG" >&2 || true
    fi
  fi

  if [[ -n "$VLLM_PID" ]]; then
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      kill -TERM -- -"$VLLM_PID" 2>/dev/null || kill "$VLLM_PID" 2>/dev/null || true
      for _ in $(seq 1 "$SHUTDOWN_TIMEOUT"); do
        kill -0 "$VLLM_PID" 2>/dev/null || break
        sleep 1
      done
    fi
    kill -KILL -- -"$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  if [[ -n "$DC_PID" ]] && kill -0 "$DC_PID" 2>/dev/null; then
    kill "$DC_PID" 2>/dev/null || true
    wait "$DC_PID" 2>/dev/null || true
  fi
  if [[ "$KEEP_VALIDATION_DIRS" == 1 || "$exit_code" != 0 ]]; then
    echo "Keeping validation directories: CACHE_DIR=$CACHE_DIR META_DIR=$META_DIR"
    echo "Logs: DISK_CACHE_LOG=$DISK_CACHE_LOG VLLM_LOG=$VLLM_LOG"
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

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

wait_for_url() {
  local url=$1
  local name=$2
  local timeout=$3
  for i in $(seq 1 "$timeout"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "==> $name ready after ${i}s"
      return 0
    fi
    if [[ "$name" == "disk-cache" ]] && [[ -n "$DC_PID" ]] && ! kill -0 "$DC_PID" 2>/dev/null; then
      echo "$name exited early" >&2
      cat "$DISK_CACHE_LOG" >&2 || true
      return 1
    fi
    if [[ "$name" == "vLLM" ]] && [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "$name exited early" >&2
      tail -200 "$VLLM_LOG" >&2 || true
      return 1
    fi
    if [[ $((i % 30)) -eq 0 && "$name" == "vLLM" ]]; then
      echo "==> still waiting for vLLM (${i}s)"
      tail -20 "$VLLM_LOG" || true
    fi
    sleep 1
  done
  echo "timed out waiting for $name at $url" >&2
  if [[ "$name" == "vLLM" ]]; then
    tail -200 "$VLLM_LOG" >&2 || true
  fi
  return 1
}

require_cmd curl
require_cmd setsid
require_cmd "$PYTHON_BIN"
require_cmd "$VLLM_BIN"
mkdir -p "$ROOT_DIR/bin" "$CACHE_DIR" "$META_DIR"

if curl -fsS --max-time 1 "$BASE_CACHE_URL/stats" >/dev/null 2>&1; then
  echo "refusing to run: disk-cache address already responds: $BASE_CACHE_URL" >&2
  exit 1
fi
if curl -fsS --max-time 1 "$BASE_VLLM_URL/v1/models" >/dev/null 2>&1; then
  echo "refusing to run: vLLM address already responds: $BASE_VLLM_URL" >&2
  exit 1
fi

cd "$ROOT_DIR"
if [[ -z "${SKIP_BUILD_DISK_CACHE:-}" ]]; then
  echo "==> Building disk-cache"
  (cd engine && go build -o "$DISK_CACHE_BIN" ./cmd/disk-cache)
fi

if [[ ! -x "$DISK_CACHE_BIN" ]]; then
  echo "disk-cache binary is not executable: $DISK_CACHE_BIN" >&2
  exit 1
fi

echo "==> Starting disk-cache on $DISK_CACHE_ADDR"
"$DISK_CACHE_BIN" \
  -listen "$DISK_CACHE_ADDR" \
  -cache-path "$CACHE_DIR" \
  -metadata-path "$META_DIR" \
  -max-size "$DISK_CACHE_MAX_SIZE" \
  >"$DISK_CACHE_LOG" 2>&1 &
DC_PID=$!
wait_for_url "$BASE_CACHE_URL/stats" disk-cache 30

if [[ "$DISABLE_PREFIX_CACHING" == 1 ]]; then
  PREFIX_CACHE_ARG=(--no-enable-prefix-caching)
else
  PREFIX_CACHE_ARG=()
fi

KV_CONFIG=$("$PYTHON_BIN" - <<'PY' "$CACHE_DIR" "$BASE_CACHE_URL" "$TARGET_DEVICE" "$CHUNK_SIZE_MB" "$STORAGE_BACKEND" "$CONNECTOR_MODULE"
import json
import sys
cache_dir, engine_addr, target_device, chunk_mb, backend, module = sys.argv[1:]
print(json.dumps({
    "kv_connector": "DiskCacheConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": module,
    "kv_connector_extra_config": {
        "disk_cache_path": cache_dir,
        "disk_cache_engine_addr": engine_addr,
        "target_device": target_device,
        "disk_cache_chunk_size_mb": int(chunk_mb),
        "storage_backend": backend,
    },
}, separators=(",", ":")))
PY
)

echo "==> Starting vLLM on $VLLM_ADDR"
set -f
# shellcheck disable=SC2086
setsid env \
  -u MODEL_PATH -u VLLM_BIN -u PYTHON_BIN -u CONNECTOR_MODULE \
  -u DISK_CACHE_ADDR -u VLLM_ADDR -u CACHE_DIR -u META_DIR \
  -u DISK_CACHE_BIN -u DISK_CACHE_MAX_SIZE -u VLLM_LOG -u DISK_CACHE_LOG \
  -u CHUNK_SIZE_MB -u STORAGE_BACKEND -u TARGET_DEVICE -u BLOCK_SIZE \
  -u REQUEST_REPETITIONS -u MAX_TOKENS -u TEMPERATURE -u READY_TIMEOUT \
  -u SHUTDOWN_TIMEOUT -u VLLM_EXTRA_ARGS -u DISABLE_PREFIX_CACHING \
  -u KEEP_VALIDATION_DIRS -u SKIP_BUILD_DISK_CACHE \
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$VLLM_BIN" serve "$MODEL_PATH" \
    --host "${VLLM_ADDR%:*}" \
    --port "${VLLM_ADDR##*:}" \
    --enable-prompt-tokens-details \
    "${PREFIX_CACHE_ARG[@]}" \
    --kv-transfer-config "$KV_CONFIG" \
    $VLLM_EXTRA_ARGS \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
set +f
wait_for_url "$BASE_VLLM_URL/v1/models" vLLM "$READY_TIMEOUT"

echo "==> Sending repeated prompt and checking disk-cache retrieval stats"
"$PYTHON_BIN" - <<'PY' "$BASE_VLLM_URL" "$BASE_CACHE_URL" "$REQUEST_REPETITIONS" "$MAX_TOKENS" "$TEMPERATURE" "$BLOCK_SIZE"
import json
import sys
import time
import urllib.request

base, stats_url, repetitions, max_tokens, temperature, block_size = sys.argv[1:]
repetitions = int(repetitions)
max_tokens = int(max_tokens)
block_size = int(block_size)
temperature = float(temperature)
stats_url = stats_url.rstrip("/") + "/stats"


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def post_json(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

model = get_json(base.rstrip("/") + "/v1/models")["data"][0]["id"]
marker = "PREDICT_VLLM_DISK_CACHE_VALIDATION"
unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau."
prompt = " ".join([marker, unit] * repetitions)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": temperature,
}

before = get_json(stats_url)
start = time.time()
first = post_json(base.rstrip("/") + "/v1/chat/completions", payload)
first_seconds = time.time() - start
mid = get_json(stats_url)
start = time.time()
second = post_json(base.rstrip("/") + "/v1/chat/completions", payload)
second_seconds = time.time() - start
after = get_json(stats_url)

def cached_tokens(resp):
    details = (resp.get("usage") or {}).get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)

first_retrieved_delta = int(mid.get("BlocksRetrieved", 0)) - int(before.get("BlocksRetrieved", 0))
second_retrieved_delta = int(after.get("BlocksRetrieved", 0)) - int(mid.get("BlocksRetrieved", 0))
second_cached_tokens = cached_tokens(second)

print("model", model)
print("before", before)
print("first_seconds", round(first_seconds, 3))
print("first_usage", first.get("usage"))
print("mid", mid)
print("second_seconds", round(second_seconds, 3))
print("second_usage", second.get("usage"))
print("after", after)
print("first_retrieved_delta", first_retrieved_delta)
print("second_retrieved_delta", second_retrieved_delta)
print("second_cached_tokens", second_cached_tokens)

if first_retrieved_delta != 0:
    raise SystemExit(f"first request unexpectedly retrieved {first_retrieved_delta} blocks")
if second_retrieved_delta <= 0:
    raise SystemExit("second request did not retrieve any disk-cache chunks")
if second_cached_tokens < block_size:
    raise SystemExit(f"second request cached_tokens too low: {second_cached_tokens}")
if int(after.get("BlocksStored", 0)) <= int(before.get("BlocksStored", 0)):
    raise SystemExit("disk-cache did not store any blocks")

print("vLLM disk-cache validation passed")
PY
