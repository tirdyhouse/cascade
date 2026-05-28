#!/root/cascade/.venv-cascade/bin/python3
"""
vLLM launch wrapper with disk-cache connector support.
Usage:
    python3 launch_vllm.py <model_path> [--kv-connector DiskCacheConnector --disk-cache-path <path>] [extra vllm args...]

If --kv-connector is provided, it configures the KVTransferConfig via --kv-transfer-config JSON
(vLLM 0.21+). Otherwise it passes all args directly to vllm.
"""
import sys
import os
import json

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
        # Inject --kv-transfer-config JSON (correct for vLLM 0.21+)
        kv_config = {
            "kv_connector": kv_connector,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "disk_cache_path": disk_cache_path,
            },
        }
        clean_args.extend(["--kv-transfer-config", json.dumps(kv_config)])
        print(f"[launch_vllm] KV connector: {kv_connector}, cache path: {disk_cache_path}", flush=True)

    print(f"[launch_vllm] Running: {VLLM_BIN} serve {' '.join(clean_args)}", flush=True)
    os.execvp(VLLM_BIN, ["vllm", "serve"] + clean_args)

if __name__ == "__main__":
    main()
