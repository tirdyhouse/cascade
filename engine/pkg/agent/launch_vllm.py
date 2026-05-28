#!/root/cascade/.venv-cascade/bin/python3
"""
vLLM launch wrapper with disk-cache connector support.
Usage:
    python3 launch_vllm.py <model_path> [--kv-connector DiskCacheConnector --disk-cache-path <path>] [extra vllm args...]

If --kv-connector is provided, it configures the KVTransferConfig programmatically
(vLLM 0.21 does not expose these as CLI args).
Otherwise it passes all args directly to vllm.
"""
import sys
import os
import subprocess

VLLM_BIN = "/root/cascade/.venv-cascade/bin/vllm"

def main():
    args = sys.argv[1:]

    # Check if we need to set up the KV connector
    kv_connector = None
    disk_cache_path = None
    clean_args = []
    i = 0
    while i < len(args):
        if args[i] == "--kv-connector" and i + 1 < len(args):
            kv_connector = args[i + 1]
            i += 2
        elif args[i] == "--disk-cache-path" and i + 1 < len(args):
            disk_cache_path = args[i + 1]
            i += 2
        else:
            clean_args.append(args[i])
            i += 1

    if kv_connector and disk_cache_path:
        # Use Python API to configure the connector
        os.environ["VLLM_KV_CONNECTOR"] = kv_connector
        os.environ["VLLM_KV_CONNECTOR_EXTRA_CONFIG"] = f'{{"disk_cache_path": "{disk_cache_path}"}}'
        print(f"[launch_vllm] KV connector: {kv_connector}, cache path: {disk_cache_path}")

    cmd = [VLLM_BIN, "serve"] + clean_args
    print(f"[launch_vllm] Running: {' '.join(cmd)}", flush=True)
    os.execvp(VLLM_BIN, ["vllm", "serve"] + clean_args)

if __name__ == "__main__":
    main()
