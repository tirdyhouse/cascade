# 共享 KV Cache 元数据服务

本文描述当前 Cascade shared-pool MVP 的部署方式。目标是让多台 GPU
机器访问同一份 KV chunk 对象，并通过一份中心 Go metadata service 得到一致的
匹配和加载结果。

## 架构与边界

~~~text
GPU node A --+
GPU node B --+-- HTTP --> disk-cache metadata service --> 本地 Pebble
GPU node N --+                                      |
                                                    +--> 校验共享 nvfile 根目录

所有 GPU node <------------------------------------------> 同一共享 KV 对象目录
~~~

- KV 对象在所有节点可见的共享目录中，例如 /mnt/nvfile/cascade/kv-v2。
- Pebble 元数据只能由中心 metadata service 打开，放在该服务机器的本地持久磁盘，
  例如 /var/lib/cascade/shared-cache-meta。不要把 Pebble 目录放到 nvfile，也不要
  每台 GPU 机器各启动一个 shared metadata service。
- 中心服务会在共享根目录创建 .cascade-shared-cache.json。vLLM connector 启动时
  必须从本机挂载读到同一文件且 cache ID 一致，否则拒绝启动。
- chunk 文件先在共享目录完整落盘并原子发布；metadata service 确认文件可见、路径
  未逃逸根目录且大小匹配后，才提交 metadata。
- 当前 shared mode禁用自动淘汰和 /evict。在读 lease 和分布式 GC 完成前，自动删除
  可能删掉另一台机器正在加载的对象。容量监控和人工清理仍需要运维流程。

此版本提供的是共享缓存数据面 MVP，不提供 metadata 高可用、跨机 cache GC、租户隔离
或跨节点 tensor parallel。

## 1. 所有节点准备共享目录

在 metadata service 所在机器和每台 GPU 节点上，确认同一份 nvfile 已挂载。
以下示例假定每台机器使用相同挂载路径；路径不同也可以，但必须挂到同一底层目录。

~~~bash
export CACHE_ROOT=/mnt/nvfile/cascade/kv-v2
export CACHE_ID=nvfile-cascade-prod-1
findmnt -T "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT"
~~~

不要预先手工创建 .cascade-shared-cache.json。它由中心 metadata service 首次启动
时以 create-if-absent 方式生成。

## 2. 启动中心 metadata service

只在一台具备共享目录挂载且本地持久盘可靠的机器上执行：

~~~bash
cd /path/to/cascade
make build-engine
mkdir -p /var/lib/cascade/shared-cache-meta
./bin/disk-cache --listen 0.0.0.0:9100 --metadata-mode shared --shared-cache-id "$CACHE_ID" --cache-path "$CACHE_ROOT" --metadata-path /var/lib/cascade/shared-cache-meta
~~~

metadata-path 是中心服务的本地目录，不是共享存储路径。shared mode 会明确关闭
自动 eviction，所以 max-size 在当前模式下不参与自动回收。

检查服务和共享根标识：

~~~bash
curl -fsS http://<metadata-host>:9100/v2/cluster/info
cat "$CACHE_ROOT/.cascade-shared-cache.json"
~~~

预期 metadata_mode 为 shared，shared_cache_id 等于 CACHE_ID，
published_object_verified 为 true，eviction_enabled 为 false。

## 3. 启动 S 端与每台 C-Agent

S 端仍然负责节点注册、心跳和命令分发：

~~~bash
cd /path/to/cascade
make build-cs
./bin/cluster-server --rpcx-port 9000 --http-port 8080
~~~

每台 GPU 节点都使用相同的 cache ID 和 metadata URL：

~~~bash
./bin/c-agent --server <cluster-server-host>:9000 --node-id gpu-node-01 --cache-mode shared_pool --cache-path http://<metadata-host>:9100 --shared-cache-root "$CACHE_ROOT" --shared-cache-id "$CACHE_ID" --gpu-type <GPU型号> --gpu-mem <显存MB> --gpu-count <GPU数量> --work-dir /var/lib/cascade/agent --vllm-host 0.0.0.0 --vllm-port 8000 --advertise-host <gpu-node-data-ip>
~~~

Agent 会把共享目录、metadata URL 和 cache ID 写入 vLLM 的
kv_connector_extra_config。S 端会拒绝注册到同一 shared-pool、但 cache ID 不同的
节点；集群汇总也会对同一共享 metadata service 的 cache 统计去重。

`--advertise-host` 必须是 S 端网关可直接访问的推理网地址，不应依赖默认的出站网卡自动探测。这样管理网、存储网与推理网分离时，网关仍会路由到正确的 vLLM endpoint。

确认注册：

~~~bash
curl -fsS http://<cluster-server-host>:8080/api/v1/cluster/status
~~~

The cluster server also exposes the first OpenAI-compatible gateway at
`http://<cluster-server-host>:8080/v1`. It routes requests only to fresh,
running replicas of the requested model. See `docs/cluster-gateway.md` for
the client endpoint and deployment constraints.

## 4. 启动带 Cascade 的 vLLM

通过 S 端下发 start_vllm 命令时，params.enable_disk_cache 必须为 true。

~~~bash
curl -fsS -X POST http://<cluster-server-host>:8080/api/v1/command -H 'Content-Type: application/json' -d '{"action":"start_vllm","target":"gpu-node-01","timeout":600,"params":{"model":"Qwen2.5-7B-Instruct-AWQ","gpu_util":"0.9","enable_disk_cache":"true"}}'
~~~

手工启动 vLLM 时，所有节点的 connector 配置必须保持以下字段一致：

~~~json
{
  "disk_cache_path": "/mnt/nvfile/cascade/kv-v2",
  "disk_cache_engine_addr": "http://<metadata-host>:9100",
  "disk_cache_shared": true,
  "disk_cache_shared_id": "nvfile-cascade-prod-1"
}
~~~

使用 Cascade 时需关闭 vLLM 内建 prefix caching，避免 GPU 内建缓存绕过外部
connector：

~~~text
--no-enable-prefix-caching
~~~

## 5. 两机验证

1. 在 node A 发送一个长 prompt，让它完成 KV 存储。
2. 在 node B 发送完全相同的 prompt。
3. 在 metadata service 查询 /stats，并查看 node B 的 vLLM 日志。

node B 的 connector 必须能通过 /v2/chunks/match 和 /v2/chunks/resolve 得到对象，
随后从本机的 CACHE_ROOT 读取同一个相对路径。若根标识文件缺失、ID 不一致、
metadata service 不处于 shared mode，node B 的 vLLM 会在启动阶段失败，而不是退化
为错误的本地缓存。

## 运维限制

- 当前 metadata service 是单实例。生产部署前需要持久化备份、主备或选主和故障恢复。
- shared mode 尚无 lease 和垃圾回收；不要在服务流量期间直接删除共享目录中的对象。
- HTTP metadata API 当前没有内建认证。必须先放在受限网络中，后续接入 mTLS/API
  鉴权。
- GDS/cuFile 只影响对象 I/O 路径；共享 metadata 协议本身不依赖 direct GDS。
