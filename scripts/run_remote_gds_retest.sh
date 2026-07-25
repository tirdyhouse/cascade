#!/usr/bin/env bash
# Run the six-way cache comparison on the configured GPU test host.
# All processes are started in their own sessions and only those recorded
# process groups are stopped. Existing services are intentionally untouched.

set -euo pipefail

VENV=${VENV:-/mnt/data/vllmtest/.venv}
MODEL_PATH=${MODEL_PATH:-/mnt/data/Qwen2.5-7B-Instruct-AWQ}
MODEL_NAME=${MODEL_NAME:-qwen25-7b}
CASCADE_ROOT=${CASCADE_ROOT:-/mnt/data/vllmtest/cascade-current}
TOOLS_DIR=${TOOLS_DIR:-/mnt/data/vllmtest/gds-retest-tools}
BENCH_SCRIPT=${BENCH_SCRIPT:-$TOOLS_DIR/benchmark_cache_ttft.py}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-$TOOLS_DIR/summarize_cache_retest.py}
CPU_CONFIG_TEMPLATE=${CPU_CONFIG_TEMPLATE:-$TOOLS_DIR/lmcache-local-cpu.yaml}
DISK_CONFIG_TEMPLATE=${DISK_CONFIG_TEMPLATE:-$TOOLS_DIR/lmcache-local-disk.yaml}
GDS_CONFIG_TEMPLATE=${GDS_CONFIG_TEMPLATE:-$TOOLS_DIR/lmcache-gds-unregistered-compat.yaml}
LMCACHE_GDS_COMPAT_DIR=${LMCACHE_GDS_COMPAT_DIR:-$TOOLS_DIR/lmcache_compat/unregistered_cufile}

RUN_ID=${RUN_ID:-gds-retest-$(date +%Y%m%d-%H%M%S)}
RESULTS_ROOT=${RESULTS_ROOT:-/mnt/data/vllmtest/gds-retest-results}
RUN_DIR=${RUN_DIR:-$RESULTS_ROOT/$RUN_ID}
CACHE_ROOT=${CACHE_ROOT:-/mnt/nvfile/codex-gds-retest/$RUN_ID}

VLLM_PORT=${VLLM_PORT:-18205}
ENGINE_PORT=${ENGINE_PORT:-19205}
GPU_UTIL=${GPU_UTIL:-0.75}
NUM_DOCUMENTS=${NUM_DOCUMENTS:-10}
DOCUMENT_TOKENS=${DOCUMENT_TOKENS:-4096}
MAX_TOKENS=${MAX_TOKENS:-10}
# Comma-separated subset for quick isolated checks; default runs all rows.
MODE_FILTER=${MODE_FILTER:-all}
# Six workers were the best stable point on the T4 + nvfile compatibility
# path: four left cuFile wait time exposed, while eight exceeded the fully
# registered double-buffer depth of the 192 MiB BAR1-safe staging pool.
CASCADE_GDS_LOAD_WORKERS=${CASCADE_GDS_LOAD_WORKERS:-6}
CASCADE_GDS_STAGING_BUFFER_MIB=${CASCADE_GDS_STAGING_BUFFER_MIB:-192}
# POSIX read-ahead depth. The historical baseline is four; larger values are
# useful once the compiled scatter makes I/O visible on the critical path.
CASCADE_POSIX_LOAD_WORKERS=${CASCADE_POSIX_LOAD_WORKERS:-4}
# Experimental layer-major injection microbatch size for the Cascade POSIX
# loader.  One retains the legacy per-object behavior.
CASCADE_POSIX_INJECT_BATCH_CHUNKS=${CASCADE_POSIX_INJECT_BATCH_CHUNKS:-4}
# Set to zero only for a controlled benchmark of the path-validation LRU.
# Production defaults to retaining verified immutable object paths.
CASCADE_RESOLVED_OBJECT_PATH_CACHE_ENTRIES=${CASCADE_RESOLVED_OBJECT_PATH_CACHE_ENTRIES:-4096}
# Cache verified immutable .cobj headers and tensor slice offsets.  This avoids
# reopening and parsing every 64 KiB header on repeated full-prefix hits.
CASCADE_OBJECT_LAYOUT_CACHE_ENTRIES=${CASCADE_OBJECT_LAYOUT_CACHE_ENTRIES:-4096}
# Optional bounded pinned-host hot tier. Keep zero for the durable POSIX row;
# set to 4 for a same-tier comparison with LMCache's 4 GiB LocalCPU backend.
CASCADE_POSIX_HOST_CACHE_GIB=${CASCADE_POSIX_HOST_CACHE_GIB:-0}
# Same-tier comparison capacity for the dedicated Cascade host-memory row.
CASCADE_HOST_CACHE_GIB=${CASCADE_HOST_CACHE_GIB:-4}
# Reserve the same order of pinned staging memory as LMCache LocalDisk's
# 0.25 GiB CPU transfer tier, but warm it at process startup so the first
# measured POSIX hit does not pay hundreds of page-pinning calls.
CASCADE_POSIX_PINNED_STAGING_MIB=${CASCADE_POSIX_PINNED_STAGING_MIB:-256}
# This benchmark validates the optimized all-layer CUDA paged-KV scatter and
# fails below if the extension cannot be loaded. Set zero only for an explicit
# Python-fallback control experiment.
CASCADE_COMPILED_TRANSFER=${CASCADE_COMPILED_TRANSFER:-1}
# Experimental: permit vLLM's default FULL_AND_PIECEWISE graph selection.
# Keep disabled for general producer workloads because save_kv_layer is a
# Python hook; the full-hit comparison has no store work during graph replay.
CASCADE_ALLOW_FULL_CUDAGRAPH=${CASCADE_ALLOW_FULL_CUDAGRAPH:-0}
# Enable only for diagnostic runs; it prints a request-level breakdown of
# resolve, disk I/O, H2D, and paged-KV injection time.
CASCADE_LOAD_PROFILE=${CASCADE_LOAD_PROFILE:-0}
# Diagnostic fairness switch.  LMCache 0.4.3 silently skips non-layerwise KV
# restore when FULL graph replay supplies no attention metadata.  Forcing
# PIECEWISE keeps its engine-level use_layerwise setting false while ensuring
# start_load_kv receives usable metadata and actually retrieves the payload.
FORCE_PIECEWISE_CUDAGRAPH=${FORCE_PIECEWISE_CUDAGRAPH:-1}
export CASCADE_GDS_LOAD_WORKERS CASCADE_GDS_STAGING_BUFFER_MIB \
  CASCADE_POSIX_LOAD_WORKERS CASCADE_POSIX_INJECT_BATCH_CHUNKS \
  CASCADE_RESOLVED_OBJECT_PATH_CACHE_ENTRIES \
  CASCADE_OBJECT_LAYOUT_CACHE_ENTRIES CASCADE_POSIX_HOST_CACHE_GIB \
  CASCADE_POSIX_PINNED_STAGING_MIB \
  CASCADE_COMPILED_TRANSFER \
  CASCADE_ALLOW_FULL_CUDAGRAPH CASCADE_LOAD_PROFILE

VLLM_PID=""
ENGINE_PID=""

source "$VENV/bin/activate"
PYTHON_BIN=$(command -v python)
VLLM_BIN=$(command -v vllm)

export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/gds/lib64:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}

# The base EL8 compiler on the test host is GCC 8, while current PyTorch CUDA
# extensions require GCC 9 or newer.  Prefer the installed toolset compiler
# when the caller did not already select one, so an enabled compiled-transfer
# benchmark cannot silently fall back to the Python scatter path.
if [ -x /opt/rh/gcc-toolset-13/root/usr/bin/gcc ] && \
   [ -x /opt/rh/gcc-toolset-13/root/usr/bin/g++ ]; then
  export CC=${CC:-/opt/rh/gcc-toolset-13/root/usr/bin/gcc}
  export CXX=${CXX:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}
fi

mkdir -p "$RUN_DIR" "$CACHE_ROOT"

stop_group() {
  local pid=${1:-}
  local pgid
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ' || true)
  if [ -n "$pgid" ] && [ "$pgid" != "$$" ]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  if [ -n "$pgid" ] && [ "$pgid" != "$$" ]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  stop_group "$VLLM_PID"
  stop_group "$ENGINE_PID"
}
trap cleanup EXIT INT TERM

assert_port_free() {
  local port=$1
  # A vLLM API process can exit slightly before its EngineCore child releases
  # the listening socket.  Persistent-backend tests restart immediately, so
  # allow that bounded teardown window before treating the port as foreign.
  for _ in $(seq 1 30); do
    if ! ss -ltn | awk 'NR > 1 {print $4}' | grep -Eq ":${port}$"; then
      return 0
    fi
    sleep 1
  done
  echo "Port ${port} is still in use after 30 seconds" >&2
  exit 1
}

wait_http() {
  local url=$1
  local pid=$2
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Test process ${pid} exited before ${url} was ready" >&2
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

wait_log() {
  local file=$1
  local pattern=$2
  for _ in $(seq 1 30); do
    if grep -aEq "$pattern" "$file" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "Expected log pattern not found: ${pattern}" >&2
  return 1
}

record_environment() {
  local cascade_commit
  {
    date --iso-8601=seconds
    printf 'cascade_root=%s\n' "$CASCADE_ROOT"
    printf 'run_id=%s\n' "$RUN_ID"
    if [ -f "$CASCADE_ROOT/GIT_COMMIT" ]; then
      printf 'cascade_commit='
      tr -d '\r\n' < "$CASCADE_ROOT/GIT_COMMIT"
      printf '\n'
    elif cascade_commit=$(git -C "$CASCADE_ROOT" rev-parse HEAD 2>/dev/null); then
      printf 'cascade_commit=%s\n' "$cascade_commit"
    else
      printf 'cascade_commit=unknown\n'
    fi
    if [ -x "$CASCADE_ROOT/bin/disk-cache" ]; then
      sha256sum "$CASCADE_ROOT/bin/disk-cache"
    fi
    printf '%s\n' \
      "mode_filter=$MODE_FILTER" \
      "num_documents=$NUM_DOCUMENTS" \
      "document_tokens=$DOCUMENT_TOKENS" \
      "max_tokens=$MAX_TOKENS" \
      "gpu_util=$GPU_UTIL" \
      "force_piecewise_cudagraph=$FORCE_PIECEWISE_CUDAGRAPH" \
      "cascade_compiled_transfer=$CASCADE_COMPILED_TRANSFER" \
      "cascade_host_cache_gib=$CASCADE_HOST_CACHE_GIB" \
      "cascade_posix_host_cache_gib=$CASCADE_POSIX_HOST_CACHE_GIB" \
      "cascade_posix_load_workers=$CASCADE_POSIX_LOAD_WORKERS" \
      "cascade_gds_load_workers=$CASCADE_GDS_LOAD_WORKERS" \
      "cascade_gds_staging_buffer_mib=$CASCADE_GDS_STAGING_BUFFER_MIB"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    findmnt -T /mnt/nvfile -o TARGET,SOURCE,FSTYPE,OPTIONS
    lsmod | grep '^nvidia_fs' || true
    cat /proc/driver/nvidia-fs/version 2>/dev/null || true
    "$VLLM_BIN" --version
    "$PYTHON_BIN" -c 'import lmcache, vllm; print("lmcache", lmcache.__file__); print("vllm", vllm.__version__)'
  } > "$RUN_DIR/environment.txt" 2>&1
  python /usr/local/cuda-13.2/gds/tools/gdscheck.py -p > "$RUN_DIR/gdscheck.txt" 2>&1 || true
}

make_cascade_config() {
  local cache_path=$1
  local engine_url=$2
  local backend=$3
  "$PYTHON_BIN" - "$cache_path" "$engine_url" "$backend" <<'PY'
import json
import os
import sys

cache_path, engine_url, backend = sys.argv[1:]
config = {
    "kv_connector": "DiskCacheConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "adapter.vllm.connector",
    "kv_connector_extra_config": {
        "disk_cache_path": cache_path,
        "disk_cache_engine_addr": engine_url,
        "target_device": "cuda",
        "disk_cache_chunk_size_tokens": 256,
        "storage_backend": backend,
        "storage_backend_strict": backend == "gds",
        "disk_cache_posix_load_workers": int(
            os.environ["CASCADE_POSIX_LOAD_WORKERS"]
        ),
        "disk_cache_posix_inject_batch_chunks": int(
            os.environ["CASCADE_POSIX_INJECT_BATCH_CHUNKS"]
        ),
        "disk_cache_resolved_object_path_cache_entries": int(
            os.environ["CASCADE_RESOLVED_OBJECT_PATH_CACHE_ENTRIES"]
        ),
        "disk_cache_object_layout_cache_entries": int(
            os.environ["CASCADE_OBJECT_LAYOUT_CACHE_ENTRIES"]
        ),
        "disk_cache_posix_host_cache_gib": float(
            os.environ["CASCADE_POSIX_HOST_CACHE_GIB"]
        ),
        "disk_cache_posix_pinned_staging_mib": float(
            os.environ["CASCADE_POSIX_PINNED_STAGING_MIB"]
        ),
        "disk_cache_compiled_transfer": os.environ[
            "CASCADE_COMPILED_TRANSFER"
        ].lower() in {"1", "true", "yes", "on"},
        "disk_cache_allow_full_cudagraph": os.environ[
            "CASCADE_ALLOW_FULL_CUDAGRAPH"
        ].lower() in {"1", "true", "yes", "on"},
        # Six workers and a 192 MiB registered pool are the best stable point
        # measured on this T4. Environment values retain isolated controls.
        "disk_cache_gds_load_workers": int(os.environ["CASCADE_GDS_LOAD_WORKERS"]),
        "disk_cache_gds_staging_buffer_mib": int(os.environ["CASCADE_GDS_STAGING_BUFFER_MIB"]),
        "disk_cache_load_profile": os.environ.get(
            "CASCADE_LOAD_PROFILE", "0"
        ).lower() in {"1", "true", "yes", "on"},
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
}

start_engine() {
  local mode_dir=$1
  local storage_path=$2
  local metadata_path=$3
  assert_port_free "$ENGINE_PORT"
  setsid "$CASCADE_ROOT/bin/disk-cache" \
    -listen "127.0.0.1:${ENGINE_PORT}" \
    -cache-path "$storage_path" \
    -metadata-path "$metadata_path" \
    -max-size 50GB > "$mode_dir/engine.log" 2>&1 &
  ENGINE_PID=$!
  printf '%s\n' "$ENGINE_PID" > "$mode_dir/engine.pid"
  wait_http "http://127.0.0.1:${ENGINE_PORT}/stats" "$ENGINE_PID"
}

start_vllm() {
  local mode_dir=$1
  local kv_config=$2
  local lmcache_config=${3:-}
  local log_name=${4:-vllm.log}
  local pid_name=${5:-vllm.pid}
  local lmcache_gds_compat=${6:-false}
  local -a compilation_args=()
  local -a lmcache_compat_env=()
  if [ "$FORCE_PIECEWISE_CUDAGRAPH" = "1" ]; then
    compilation_args=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
  fi
  if [ "$lmcache_gds_compat" = "true" ]; then
    if [ ! -f "$LMCACHE_GDS_COMPAT_DIR/sitecustomize.py" ]; then
      echo "LMCache GDS compatibility shim is missing: $LMCACHE_GDS_COMPAT_DIR/sitecustomize.py" >&2
      return 1
    fi
    lmcache_compat_env=(
      "PYTHONPATH=$LMCACHE_GDS_COMPAT_DIR:${PYTHONPATH:-}"
      "LMCACHE_UNREGISTERED_CUFILE=1"
    )
  fi
  assert_port_free "$VLLM_PORT"
  if [ -n "$lmcache_config" ]; then
    setsid env \
      "${lmcache_compat_env[@]}" \
      LMCACHE_CONFIG_FILE="$lmcache_config" \
      PYTHONHASHSEED="$PYTHONHASHSEED" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      "$VLLM_BIN" serve "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --host 127.0.0.1 --port "$VLLM_PORT" \
      --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$GPU_UTIL" --no-enable-prefix-caching \
      --kv-transfer-config "$kv_config" "${compilation_args[@]}" \
      > "$mode_dir/$log_name" 2>&1 &
  else
    setsid env \
      PYTHONPATH="$CASCADE_ROOT:${PYTHONPATH:-}" \
      PYTHONHASHSEED="$PYTHONHASHSEED" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      "$VLLM_BIN" serve "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --host 127.0.0.1 --port "$VLLM_PORT" \
      --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$GPU_UTIL" --no-enable-prefix-caching \
      --kv-transfer-config "$kv_config" "${compilation_args[@]}" \
      > "$mode_dir/$log_name" 2>&1 &
  fi
  VLLM_PID=$!
  printf '%s\n' "$VLLM_PID" > "$mode_dir/$pid_name"
  wait_http "http://127.0.0.1:${VLLM_PORT}/v1/models" "$VLLM_PID"
}

run_phase() {
  local mode_dir=$1
  local phase=$2
  local output_stem=${3:-$phase}
  local -a expected_args=()
  if [ "$phase" = "query" ] && [ -f "$mode_dir/warmup.json" ]; then
    expected_args=(--expected-output "$mode_dir/warmup.json")
  fi
  "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --phase "$phase" --host 127.0.0.1 --port "$VLLM_PORT" \
    --model "$MODEL_NAME" --num-documents "$NUM_DOCUMENTS" \
    --document-tokens "$DOCUMENT_TOKENS" --max-tokens "$MAX_TOKENS" \
    --output "$mode_dir/${output_stem}.json" "${expected_args[@]}" \
    > "$mode_dir/${output_stem}.log" 2>&1
}

capture_evidence() {
  local mode_dir=$1
  local log_name=${2:-vllm.log}
  local append=${3:-false}
  if [ "$append" = "true" ]; then
    grep -aE 'Storage backend:|backend=|Created backend:|GDS backend using fstype|Using cufile|No base pointer found|LMCacheEngine marked as init failed|cuFileBufRegister|Registered .* staging pool|GDS staging pool|GDS read-ahead' \
      "$mode_dir/$log_name" >> "$mode_dir/backend-evidence.txt" || true
  else
    grep -aE 'Storage backend:|backend=|Created backend:|GDS backend using fstype|Using cufile|No base pointer found|LMCacheEngine marked as init failed|cuFileBufRegister|Registered .* staging pool|GDS staging pool|GDS read-ahead' \
      "$mode_dir/$log_name" > "$mode_dir/backend-evidence.txt" || true
  fi
}

mode_selected() {
  local mode=$1
  if [ "$MODE_FILTER" = "all" ]; then
    return 0
  fi
  case ",$MODE_FILTER," in
    *",$mode,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_cascade() {
  local mode=$1
  local backend=$2
  local host_cache_gib=${3:-$CASCADE_POSIX_HOST_CACHE_GIB}
  local mode_dir="$RUN_DIR/$mode"
  local cache_path="$CACHE_ROOT/$mode/storage"
  local metadata_path="$CACHE_ROOT/$mode/metadata"
  local kv_config
  mkdir -p "$mode_dir" "$cache_path" "$metadata_path"
  start_engine "$mode_dir" "$cache_path" "$metadata_path"
  local saved_host_cache_gib=$CASCADE_POSIX_HOST_CACHE_GIB
  CASCADE_POSIX_HOST_CACHE_GIB=$host_cache_gib
  kv_config=$(make_cascade_config "$cache_path" "http://127.0.0.1:${ENGINE_PORT}" "$backend")
  CASCADE_POSIX_HOST_CACHE_GIB=$saved_host_cache_gib
  start_vllm "$mode_dir" "$kv_config"
  if [ "$backend" = "gds" ]; then
    wait_log "$mode_dir/vllm.log" 'NvFileBackend|GDS binding'
  else
    wait_log "$mode_dir/vllm.log" 'PosixBackend'
  fi
  if [[ "${CASCADE_COMPILED_TRANSFER,,}" =~ ^(1|true|yes|on)$ ]]; then
    wait_log "$mode_dir/vllm.log" \
      'DiskCacheConnector v2 ready:.*compiled_transfer=True'
  fi
  capture_evidence "$mode_dir"
  curl -fsS "http://127.0.0.1:${ENGINE_PORT}/stats" > "$mode_dir/stats-before-warmup.json"
  run_phase "$mode_dir" warmup
  sleep 8
  curl -fsS "http://127.0.0.1:${ENGINE_PORT}/stats" > "$mode_dir/stats-before-query.json"
  run_phase "$mode_dir" query
  sleep 2
  curl -fsS "http://127.0.0.1:${ENGINE_PORT}/stats" > "$mode_dir/stats-after-query.json"
  stop_group "$VLLM_PID"
  VLLM_PID=""
  stop_group "$ENGINE_PID"
  ENGINE_PID=""
}

run_lmcache() {
  local mode=$1
  local config_path=$2
  local expected_backend=$3
  local mode_dir="$RUN_DIR/$mode"
  local kv_config='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  mkdir -p "$mode_dir"
  start_vllm "$mode_dir" "$kv_config" "$config_path"
  wait_log "$mode_dir/vllm.log" "Created backend: ${expected_backend}"
  capture_evidence "$mode_dir"
  run_phase "$mode_dir" warmup
  sleep 8
  wc -l < "$mode_dir/vllm.log" > "$mode_dir/query-log-start-line.txt"
  run_phase "$mode_dir" query
  sleep 2
  stop_group "$VLLM_PID"
  VLLM_PID=""
}

run_lmcache_persistent() {
  local mode=$1
  local config_path=$2
  local expected_backend=$3
  local mode_dir="$RUN_DIR/$mode"
  local kv_config='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  mkdir -p "$mode_dir"

  # Populate the persistent file backend, then discard the vLLM process so
  # query traffic cannot use any in-process state from the producer.
  start_vllm "$mode_dir" "$kv_config" "$config_path" vllm.log vllm.pid true
  wait_log "$mode_dir/vllm.log" "Created backend: ${expected_backend}"
  capture_evidence "$mode_dir"
  run_phase "$mode_dir" warmup
  sleep 8
  stop_group "$VLLM_PID"
  VLLM_PID=""

  start_vllm "$mode_dir" "$kv_config" "$config_path" \
    restart-vllm.log restart-vllm.pid true
  wait_log "$mode_dir/restart-vllm.log" "Created backend: ${expected_backend}"
  capture_evidence "$mode_dir" restart-vllm.log true
  wc -l < "$mode_dir/restart-vllm.log" > "$mode_dir/restart-query-log-start-line.txt"
  run_phase "$mode_dir" query restart-query
  sleep 2
  stop_group "$VLLM_PID"
  VLLM_PID=""
}

record_environment
cp "$DISK_CONFIG_TEMPLATE" "$RUN_DIR/lmcache-local-disk.yaml"
sed -i "s|^local_disk:.*|local_disk: \"file://$CACHE_ROOT/lmcache-local-disk\"|" \
  "$RUN_DIR/lmcache-local-disk.yaml"
cp "$GDS_CONFIG_TEMPLATE" "$RUN_DIR/lmcache-gds.yaml"
sed -i "s|^gds_path:.*|gds_path: \"$CACHE_ROOT/lmcache-gds\"|" "$RUN_DIR/lmcache-gds.yaml"

if mode_selected cascade-host; then
  run_cascade cascade-host posix "$CASCADE_HOST_CACHE_GIB"
fi
if mode_selected cascade-posix; then
  run_cascade cascade-posix posix
fi
if mode_selected cascade-gds; then
  run_cascade cascade-gds gds
fi
if mode_selected lmcache-local-cpu; then
  run_lmcache lmcache-local-cpu "$CPU_CONFIG_TEMPLATE" LocalCPUBackend
fi
if mode_selected lmcache-local-disk; then
  # LocalDiskBackend's in-memory key index is deliberately not rebuilt from
  # its .pt files on restart (its source even carries a TODO for metadata
  # recovery).  A restarted query is therefore a cache miss, not a file-cache
  # measurement.  Restricting both store/retrieve locations to LocalDiskBackend
  # still makes this a strict POSIX-file hit comparison with Cascade.
  run_lmcache lmcache-local-disk \
    "$RUN_DIR/lmcache-local-disk.yaml" LocalDiskBackend
fi
if mode_selected lmcache-gds; then
  run_lmcache_persistent lmcache-gds "$RUN_DIR/lmcache-gds.yaml" GdsBackend
fi

if [ "$MODE_FILTER" = "all" ]; then
  "$PYTHON_BIN" "$SUMMARY_SCRIPT" --run-dir "$RUN_DIR" > "$RUN_DIR/summary.log" 2>&1
fi
printf 'RESULTS_DIR=%s\n' "$RUN_DIR"
