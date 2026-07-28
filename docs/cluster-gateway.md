# Cluster Gateway

`cluster-server` now provides the client-facing inference endpoint. Clients
send a normal OpenAI-style request to one address; the server chooses a vLLM
replica instead of requiring a node ID.

## Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
```

The request and response bodies are passed through to vLLM. Streaming SSE is
copied as it arrives. Responses include `X-Cascade-Node-ID` and
`X-Cascade-Model` for operational debugging.

## Scheduling

A node is eligible when its heartbeat is fresh (15 seconds), its vLLM process
reports `running`, and its loaded model matches the request model. Local
absolute model paths and their basename are treated as equivalent so a client
can request `Qwen2.5-7B-Instruct` when an agent launched
`/models/Qwen2.5-7B-Instruct`.

The scheduler prioritizes, in order:

1. Reported vLLM queue length plus requests accepted by this gateway.
2. Reported GPU utilization.
3. Historical cache hit rate as a small tie-breaker.
4. Round-robin tie-breaking among equivalent nodes.

`--gateway-max-inflight-per-node` defaults to `16`. When every eligible node
reaches that limit, the gateway returns OpenAI-style HTTP `429` with
`Retry-After: 1`. A model with no healthy running replica returns HTTP `503`.

The cache term is deliberately not described as a prompt match: prompt-level
cache affinity needs a stable token-hash query protocol and remains future
work. A shared cache pool can already be loaded from any eligible node.

## Start

Start the control plane and gateway once:

```bash
./bin/cluster-server \
  --rpcx-port 9000 \
  --http-port 8080 \
  --gateway-max-inflight-per-node 16
```

Every GPU node must advertise an endpoint the cluster server can reach. The
controlled `start_vllm` path passes `--host` and `--port` to vLLM. Set
`--advertise-host` explicitly whenever management, storage, and inference
traffic use different network interfaces; it is the address the gateway will
actually dial.

```bash
./bin/c-agent \
  --server <cluster-server-host>:9000 \
  --node-id gpu-node-01 \
  --vllm-host 0.0.0.0 \
  --vllm-port 8000 \
  --advertise-host <gpu-node-data-ip> \
  --vllm-path /root/cascade/.venv-cascade/bin/vllm \
  --diagnostics-host 0.0.0.0 \
  --diagnostics-port 9002 \
  --cache-mode shared_pool \
  --cache-path http://<metadata-host>:9100 \
  --shared-cache-root /mnt/nvfile/cascade/kv-v2 \
  --shared-cache-id nvfile-cascade-prod-1
```

The control plane accepts only structured lifecycle parameters. It does not
accept arbitrary `raw_args`; this prevents one Agent from receiving a launch
configuration that cannot be diagnosed or reproduced by the rest of the
cluster.

From the control-plane host, verify the advertised data address before issuing
the first model start:

```bash
curl -fsS http://<cluster-server-host>:8080/api/v1/nodes/gpu-node-01
```

The response must show `info.ip` as `<gpu-node-data-ip>` and `vllm_port` as
`8000`. The Agent diagnostics endpoint must also be reachable from the control
plane if operators need remote vLLM logs.

## Client Check

After one or more agents report a running model:

```bash
curl -fsS http://<cluster-server-host>:8080/v1/models

curl -N -X POST http://<cluster-server-host>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Current Boundaries

- The gateway has no authentication or TLS termination. Keep its HTTP port on
  a private network or place an authenticated TLS proxy in front of it.
- The agent starts vLLM on `0.0.0.0` by default so the gateway can reach it.
  Firewall the vLLM port to the cluster-server network; do not expose it to
  untrusted clients.
- It does not automatically retry a failed POST. A transport failure can be
  ambiguous because the selected vLLM may already have started inference.
- The metadata service and cluster server are single instances. HA, leader
  election, durable command state, read leases, and distributed cache GC are
  separate follow-up work.
- This routes independent vLLM replicas. It does not provide cross-node tensor
  parallelism or pipeline parallelism.
