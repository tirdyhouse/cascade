# 集群磁盘 KV 缓存 · 设计文档

> 目标：为 GPU 集群提供廉价的大容量 KV 缓存层，降低 LLM 推理的显存需求。

---

## 一、产品定位

### 一句话

> 给 vLLM / SGLang 等推理框架加一个**集群磁盘缓存**，让 GPU 可以直接读写远端 NVMe SSD，
> 把 KV cache 从"显存够不够"变成"磁盘够不够"。

### 解决什么问题

```
现状:
  GPU 显存 (80GB HBM) ← KV cache 只能放这里 → 上下文一大就 OOM
                                                   ↓
                                              买更多 GPU 💸

我们的方案:
  GPU 显存 (80GB HBM)  ← 热数据
       ↓
  集群 NVMe SSD (多台)  ← 冷数据 (GDS 直读直写)
       ↓
  客户端               ← 推理请求不爆显存 ✅
```

### 和 Mooncake 的本质区别

```
Mooncake:  传输工具 —— 你定好 prefill/decode 角色，我只管传

我们:      集群缓存操作系统 —— 我帮你定角色、管存储、做调度
          + 磁盘缓存层（Mooncake 没有）
          + 冷热数据自动迁移
          + 纯磁盘节点（没 GPU，成本低）
```

### 目标客户

| 客户 | 痛点 | 我们的价值 |
|------|------|----------|
| GPU 厂商（NVIDIA/AMD/昇腾） | 客户嫌显存贵 | 让他们的 GPU 能接磁盘，降低总成本 |
| 模型供应商（卖 API 的） | 长上下文推理成本高 | 显存省 50-80%，同等 GPU 跑更多请求 |
| 云推理平台 | 客户要 1M token 上下文 | 磁盘换显存，不升级 GPU 也能跑 |

---

## 二、集群架构

### 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        框架适配器（Python, ~400 行）               │
│                                                                  │
│  ┌────────────────────────┐    ┌────────────────────────┐       │
│  │ vLLM KVConnectorBase   │    │ SGLang BaseKVConnector │       │
│  │ 实现                   │    │ 实现                   │       │
│  │                        │    │                        │       │
│  │ ▲ 本地调度决策          │    │ ▲ 本地调度决策          │       │
│  │ │ • get_num_new...     │    │ │ • ...                │       │
│  │ │ • request_finished() │    │ │                      │       │
│  │ │                      │    │ │                      │       │
│  │ ▼ 本机状态上报          │    │ ▼ 本机状态上报          │       │
│  │ │ • gpu_util, disk,   │    │ │ • ...                │       │
│  │ │   queue_len, role    │    │ │                      │       │
│  │ │                      │    │ │                      │       │
│  │ ▼ 执行存储操作          │    │ ▼ 执行存储操作          │       │
│  │   save_kv_layer()      │    │   save_kv_layer()      │       │
│  │   start_load_kv()      │    │   start_load_kv()      │       │
│  └────────┬───────────────┘    └────────┬───────────────┘       │
│           │                            │                        │
│           └──────────────┬─────────────┘                        │
│                          │ cgo / ctypes                         │
├──────────────────────────┼──────────────────────────────────────┤
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Go DiskCacheEngine (每节点一个)             │  │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ BlockManager  │  │  Eviction    │  │ Local        │  │   │
│  │  │ (rocksdb)     │  │  Policy     │  │ Scheduler    │  │   │
│  │  │ 引用计数/哈希   │  │  LRU/LFU    │  │ 本地缓存决策  │  │   │
│  │  │ 块位置映射      │  │  分层淘汰    │  │ 预取/降级    │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │   │
│  │         │                │                             │   │
│  │         └────────┬───────┘                             │   │
│  │                  │                                     │   │
│  │  ┌───────────────▼──────────────────────────────┐     │   │
│  │  │            Storage Layer                      │     │   │
│  │  │  L0:GPU  L1:DRAM  L2:NVMe  L3:Remote         │     │   │
│  │  │  GDS直读写 / RDMA传输 / io_uring              │     │   │
│  │  └───────────────┬──────────────────────────────┘     │   │
│  └──────────────────┼────────────────────────────────────┘   │
│                     │                                        │
├─────────────────────┼────────────────────────────────────────┤
│                     ▼                                        │
│  ┌──────────────────────────────────────┐                   │
│  │  Go ClusterManager (独立进程)         │                   │
│  │                                      │                   │
│  │  ● etcd 节点注册发现                  │                   │
│  │  ● 角色分配 (prefill/decode/storage)  │                   │
│  │  ● 负载感知动态调度                    │                   │
│  │  ● 故障转移 / 数据迁移                │                   │
│  │  ● 集群元数据管理                     │                   │
│  └──────────────────────────────────────┘                   │
│                     │                                        │
├─────────────────────┼────────────────────────────────────────┤
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 C 存储原语                             │   │
│  │    ┌─────────┐  ┌─────────┐  ┌────────┐  ┌────────┐ │   │
│  │    │ GDS     │  │ RDMA    │  │io_uring│  │  SPDK  │ │   │
│  │    │(cufile) │  │(ibverbs)│  │        │  │(未来)  │ │   │
│  │    └─────────┘  └─────────┘  └────────┘  └────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 三种节点角色

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Prefill 节点    │    │   Decode 节点     │    │   Storage 节点    │
│                  │    │                  │    │                  │
│ GPU: H100 x8    │    │ GPU: L40S x8    │    │ GPU: ❌          │
│ 显存: 80GB      │    │ 显存: 48GB      │    │ NVMe: 30TB x8   │
│ NVMe: 2TB       │    │ NVMe: 2TB       │    │ CPU: 随便        │
│                  │    │                  │    │                  │
│ 角色: 算 KV      │    │ 角色: 用 KV      │    │ 角色: 存 KV      │
│ MLA Prefill     │    │ MLA Decode      │    │ 冷数据仓库       │
│ 算完→存本地SSD   │    │ 缺 KV→集群拉    │    │ RDMA 供数据      │
│ 通知集群"我有"   │    │ 本地 SSD 缓存    │    │ 成本极低         │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### 请求全生命周期

```
客户端请求 → 负载均衡器
    │
    ▼
分配给 Decode 节点
    │
    ▼
Decode 节点的 KVConnector.get_num_new_matched_tokens(request)
    │
    ├── 请求的 prefix hash 在本地 SSD 有吗？
    │   └─ ✅ 有 → 直接跳过 prefill，开始 decode（最快路径）
    │
    └── 本地没有 → 查集群元数据（ClusterManager）
        │
        ├── 其他节点的 SSD 有吗？
        │   └─ ✅ 有 → RDMA 拉过来，同时存本地 SSD 做缓存
        │
        └── 都没有 → 转发到 Prefill 节点
            │
            ▼
        Prefill 节点:
            1. 算 KV cache（MLA prefill）
            2. 存本地 SSD（GDS 直写）
            3. 通知集群"我有这些 block"
            4. KV 传输给 Decode 节点（RDMA）
            │
            ▼
Decode 节点拿到 KV → decode → 输出 token
    │
    ▼
请求结束 → KVConnector.request_finished()
    ├── prefill 节点: KV 保留在 SSD，供其他 decode 复用
    └── decode 节点: 本地缓存保留一段时间
```

---

## 三、ClusterManager 设计

### ClusterManager — 全局大脑

独立进程，一个集群只跑一个。通过 etcd 选主实现高可用。

```go
// cluster/manager.go

type ClusterManager struct {
    etcd  *clientv3.Client
    nodes map[string]*NodeInfo
    meta  *MetaStore
}

type NodeInfo struct {
    ID          string
    Addr        string     // gRPC 地址
    GPUType     string     // H100 / L40S / A100
    GPUMem      int64      // 显存总量 (GB)
    GPUUtil     float64    // 当前显存利用率
    DiskSize    int64      // NVMe 总量 (GB)
    DiskFree    int64      // NVMe 剩余 (GB)
    QueueLen    int        // 等待队列长度
    CurrentRole NodeRole   // 当前角色
}

type NodeRole string

const (
    RolePrefill NodeRole = "prefill"   // 专做 prefill
    RoleDecode  NodeRole = "decode"    // 专做 decode
    RoleStorage NodeRole = "storage"   // 纯磁盘节点
    RoleHybrid  NodeRole = "hybrid"    // prefill + decode 混用
)
```

### 核心调度逻辑

```go
// 角色分配 — 根据硬件和负载自动决定节点该做什么
func (m *ClusterManager) DecideRole(node *NodeInfo) NodeRole {
    // 没 GPU → 只能做 storage
    if node.GPUMem == 0 {
        return RoleStorage
    }

    // 大显存 + 低负载 → prefill（算得快）
    if node.GPUMem > 80*GB && node.GPUUtil < 0.6 {
        return RolePrefill
    }

    // 集群 prefill 队列太长 → 把部分节点转 prefill
    if m.PrefillQueueLength() > 50 {
        return RolePrefill
    }

    // 默认做 decode
    return RoleDecode
}

// 负载均衡 — 定期重新平衡
func (m *ClusterManager) Rebalance(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    for {
        select {
        case <-ticker.C:
            for _, node := range m.nodes {
                newRole := m.DecideRole(node)
                if newRole != node.CurrentRole {
                    m.grpc.Send(node.Addr, &RoleChange{
                        NodeID:  node.ID,
                        NewRole: newRole,
                    })
                }
            }
        case <-ctx.Done():
            return
        }
    }
}

// 故障转移 — 节点挂了，恢复它的 KV 数据
func (m *ClusterManager) HandleNodeFailure(deadNodeID string) {
    blocks := m.meta.GetBlocksOnNode(deadNodeID)
    for _, block := range blocks {
        // 找持有副本的其他节点
        replicas := m.meta.GetReplicas(block.Key, deadNodeID)
        if len(replicas) > 0 {
            // 通知副本节点升级提供数据
            m.grpc.Send(replicas[0].NodeID, &PromoteReplica{
                Key: block.Key,
            })
        } else {
            // 没有副本 → 标记为丢失，prefill 节点会重新算
            m.meta.MarkLost(block.Key)
        }
    }
}
```

### gRPC 接口

```protobuf
service ClusterManager {
    // 节点 → ClusterManager（上报）
    rpc Register(NodeInfo) returns (RegisterAck);
    rpc ReportStatus(StatusReport) returns (StatusAck);
    rpc ReportNewBlocks(NewBlocks) returns (Ack);

    // 节点 → ClusterManager（查询）
    rpc LookupBlocks(BlockQuery) returns (BlockLocations);
    rpc RequestBlocks(BlockRequest) returns (TransferPlan);
    
    // ClusterManager → 节点（指令）
    rpc SetRole(RoleChange) returns (Ack);
    rpc PromoteReplica(PromoteReplicaReq) returns (Ack);
    rpc MigrateData(MigrationReq) returns (Ack);
}

message StatusReport {
    string node_id = 1;
    NodeRole role = 2;
    double gpu_util = 3;      // 0.0 - 1.0
    int64 disk_free_bytes = 4;  // 剩余磁盘
    int32 queue_len = 5;       // 等待队列
    double avg_prefill_latency_ms = 6;  // 近期平均 prefill 延迟
    double avg_decode_latency_ms = 7;   // 近期平均 decode 延迟
}
```

### etcd 数据模型

```
/disk-cache/
├── cluster/
│   ├── leader              # ClusterManager 选主
│   └── config              # 集群配置
│
├── nodes/
│   ├── {node_id}/
│   │   ├── info            # 硬件信息（静态）
│   │   ├── status          # 运行时状态（TTL，5s 过期）
│   │   └── role            # 当前角色（ClusterManager 写入）
│   └── ...
│
├── blocks/
│   └── {hash_prefix}/
│       └── {block_hash}    # KV block 元数据
│           ├── size        # 大小
│           ├── locations   # 存在哪些节点
│           └── created_at  # 创建时间
│
└── events/                 # 集群事件日志
```

---

## 四、KVConnector 设计（Python 层）

### 为什么调度逻辑放在 connector 里

vLLM 的 KVConnectorBase 提供了调度器侧的钩子，connector 可以在这些钩子里做**本地调度决策**：

```python
class KVConnectorBase(ABC):
    # ── 调度器侧钩子（决定"要做什么"）──
    @abstractmethod
    def get_num_new_matched_tokens(self, request) -> int:
        """返回集群已有多少个 token 的 KV 缓存，避免重复 prefill"""
        pass

    @abstractmethod
    def request_finished(self, request, blocks) -> CanFree:
        """请求结束，决定 KV 怎么处理"""
        pass

    @abstractmethod
    def take_events(self) -> list:
        """返回 KV 事件（新增/删除的 block），给调度器更新状态"""
        pass

    # ── Worker 侧钩子（实际"执行"）──
    @abstractmethod
    def start_load_kv(self, blocks):
        """开始加载 KV"""
        pass

    @abstractmethod
    def save_kv_layer(self, layer, kv_tensor):
        """保存当前层的 KV"""
        pass
```

### DiskCacheConnector 实现

```python
# adapter/vllm_connector.py — ~400 行

import ctypes, threading, time, logging
from typing import Optional
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase
from vllm.v1.request import Request
import grpc

# 加载 Go 编译的 .so
lib = ctypes.CDLL("libdiskcache.so")

class DiskCacheConnector(KVConnectorBase):
    """
    跑在每个 vLLM 进程内，既是本地调度器也是存储执行器。
    
    三层职责：
    1. 向 ClusterManager 上报本机状态
    2. 在 vLLM 调度器钩子里做本地缓存决策
    3. 实际执行 GDS/RDMA 存储操作
    """

    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        self.node_id = get_hostname()
        self.role = NodeRole.PREFILL  # ClusterManager 后续会改
        
        # 初始化 Go 引擎
        lib.Init(vllm_config.disk_cache_path, vllm_config.disk_cache_size)
        
        # 连接 ClusterManager
        self.cm_channel = grpc.insecure_channel(vllm_config.cluster_manager_addr)
        self.cm_stub = ClusterManagerStub(self.cm_channel)
        
        # 注册到集群
        self._register()
        
        # 定时上报状态
        self._stop_reporting = threading.Event()
        self._report_thread = threading.Thread(target=self._periodic_report)
        self._report_thread.start()

    def _periodic_report(self):
        """⏫ 每 5 秒上报本机状态给 ClusterManager"""
        while not self._stop_reporting.is_set():
            status = StatusReport(
                node_id=self.node_id,
                role=self.role,
                gpu_util=_get_gpu_util(),
                disk_free_bytes=lib.GetDiskFree(),
                queue_len=lib.GetQueueLength(),
            )
            ack = self.cm_stub.ReportStatus(status)
            if ack.HasField("role_change"):
                self.role = ack.role_change.new_role
                logging.info(f"Role changed to {self.role}")
            time.sleep(5)

    # ─── 调度器侧：本地缓存决策 ───

    def get_num_new_matched_tokens(self, request: Request) -> int:
        """🔍 检查这个请求有多少 KV 已经缓存了"""
        prefix_hash = request.prefix_hash
        
        # 1. 查本地 SSD（最快）
        n = lib.CheckLocal(prefix_hash)
        if n > 0:
            return n
        
        # 2. 查集群
        return lib.CheckRemote(prefix_hash)

    def request_finished(self, request: Request, blocks) -> bool:
        """✅ 请求结束，决定 KV 怎么处理"""
        if self.role == NodeRole.PREFILL:
            # prefill 节点：KV 存 SSD，通知集群
            lib.StoreBlocks(blocks)
            lib.NotifyCluster(blocks)
            
        elif self.role == NodeRole.DECODE:
            # decode 节点：本地 SSD 缓存一份
            lib.MaybeCache(blocks)
            
        # 返回 True = connector 自己负责释放
        # 返回 False = vLLM 调度器正常释放
        return self.role == NodeRole.PREFILL

    # ─── Worker 侧：存储执行 ───

    def save_kv_layer(self, layer, kv_tensor, *args, **kwargs):
        """💾 实际存 KV cache（GDS 直写 NVMe）"""
        lib.StoreLayer(
            layer.block_hash,
            kv_tensor.data_ptr(),   # GPU tensor 指针
            kv_tensor.element_size() * kv_tensor.numel(),
        )

    def start_load_kv(self, blocks, *args, **kwargs):
        """📖 实际取 KV cache"""
        for b in blocks:
            if lib.ExistsLocal(b.block_hash):
                # 本地 SSD 有 → GDS 直读
                lib.ReadLocal(b.block_hash, b.tensor.data_ptr(), b.tensor.numel())
            else:
                # 本地没有 → 从集群拉（RDMA）
                loc = lib.FindRemote(b.block_hash)
                if loc:
                    lib.ReadRemote(loc, b.tensor.data_ptr())
```

---

## 五、Go 引擎核心设计

### 5.1 Core CacheEngine 接口

```go
// engine/engine.go

type CacheEngine interface {
    // 本地存储操作
    StoreLayer(ctx context.Context, key *CacheKey, gpuPtr uintptr, size int64) error
    ReadLocal(ctx context.Context, key *CacheKey, gpuPtr uintptr) (int64, error)
    ReadRemote(ctx context.Context, loc *BlockLocation, gpuPtr uintptr) (int64, error)
    Remove(ctx context.Context, key *CacheKey) error

    // 元数据查询
    CheckLocal(prefixHash uint64) int64           // 返回本地缓存中连续的 token 数
    CheckRemote(prefixHash uint64) int64          // 返回集群中连续的 token 数
    ExistsLocal(key *CacheKey) bool
    FindRemote(key *CacheKey) *BlockLocation

    // 集群通信
    NotifyCluster(blocks []*CacheKey) error
    RequestBlocks(keys []*CacheKey) ([]BlockLocation, error)
    ReportStatus() error

    // 状态
    GetDiskFree() int64
    GetQueueLength() int32
}

type CacheKey struct {
    Hash    uint64 // 内容哈希（vLLM block hash）
    GroupID uint32 // 请求分组
    BlockID uint32 // 块序号
    Size    int64  // 大小（用于淘汰决策）
}

type BlockLocation struct {
    NodeID    string
    Transport TransportType // rdma | tcp
    Addr      string        // 远端地址
    Offset    int64
    Size      int64
}
```

### 5.2 淘汰策略

```go
// eviction/policy.go

type Policy interface {
    Record(key uint64, size int64, tier Tier)   // 访问/写入时调用
    Evict(targetBytes int64, fromTier Tier) []uint64 // 返回要淘汰的 key
    Remove(key uint64)                         // 删除记录
    Len() int
}

type Tier int

const (
    TierGPU  Tier = 0  // 显存（最贵）
    TierDRAM Tier = 1  // CPU 内存
    TierNVMe Tier = 2  // 本地 SSD（我们的核心）
    TierRemote Tier = 3 // 集群存储
)
```

**分层淘汰策略：**

```go
// eviction/tiered.go

type TieredPolicy struct {
    policies map[Tier]Policy  // 每层独立的淘汰策略
}

func (p *TieredPolicy) Evict(tier Tier, targetBytes int64) []uint64 {
    keys := p.policies[tier].Evict(targetBytes)
    for _, key := range keys {
        // 从当前层移到下一层（不是删除）
        lib.Migrate(key, tier+1)
    }
    return keys
}
```

### 5.3 存储分层

```
                   容量        延迟       带宽
L0: GPU 显存       ~80GB      ~1μs     2000 GB/s   ← 热数据
  │ evict (LRU)                                        vLLM block 级别
L1: CPU DRAM       ~1TB       ~100ns    50 GB/s    ← 缓冲
  │ evict (LRU)                                        PyTorch tensor
L2: 本地 NVMe SSD  ~2-30TB    ~10μs     7 GB/s     ← 🎯 核心
  │ evict (LRU+大小感知)                               GDS 直写
L3: 集群 Storage   ~无限       ~100μs    25 GB/s    ← 冷数据
  └─ 真正删除（不再降级）                               RDMA 传输
```

---

## 六、存储层设计

### 6.1 支持的后端

| 后端 | 技术 | 适用场景 | 参考来源 |
|------|------|---------|---------|
| **GDS** | NVIDIA cufile | GPU→NVMe 直读直写 | LMCache GdsBackend |
| **RDMA** | libibverbs | 跨节点传输 | Mooncake Transfer Engine |
| **io_uring** | Linux 异步 I/O | 通用磁盘读写 | Linux 内核 |
| **POSIX** | read/write | 兼容模式 | LMCache LocalDiskBackend |
| **SPDK** | 用户态 NVMe 驱动 | 极致性能（未来） | SPDK 社区 |

### 6.2 CGo 封装

```go
// storage/gds.go
// Go 调用 NVIDIA GPUDirect Storage

/*
#cgo LDFLAGS: -lcufile
#include <cufile.h>
#include <fcntl.h>

static int gds_write(const char *path, const void *gpuPtr,
                     size_t size, CUfileHandle_t *handle) {
    CUfileDescr_t desc = {
        .type = CU_FILE_HANDLE_TYPE_OPAQUE,
    };
    desc.handle.fd = open(path, O_CREAT|O_RDWR, 0644);
    if (desc.handle.fd < 0) return -1;
    
    CUresult err = cuFileHandleRegister(handle, &desc);
    if (err != CUDA_SUCCESS) return -1;
    
    return cuFileWrite(*handle, gpuPtr, size, 0, 0);
}
*/
import "C"
import "unsafe"

type GDSBackend struct{}

func (b *GDSBackend) Write(path string, gpuPtr unsafe.Pointer, size int64) error {
    var handle C.CUfileHandle_t
    cPath := C.CString(path)
    defer C.free(unsafe.Pointer(cPath))
    
    ret := C.gds_write(cPath, gpuPtr, C.size_t(size), &handle)
    if ret < 0 {
        return fmt.Errorf("GDS write failed: %d", ret)
    }
    return nil
}

// Read: 同理，cuFileRead 直接从 NVMe → GPU
func (b *GDSBackend) Read(path string, gpuPtr unsafe.Pointer, size int64) error {
    // ...
}
```

---

## 七、框架适配器

### vLLM

```python
# adapter/vllm_connector.py — ~400 行
# 见第四章完整实现

# 用法:
# vllm serve model \
#     --kv-connector disk-cache \
#     --disk-cache-path /mnt/nvme/kv-cache \
#     --disk-cache-size 2TB \
#     --cluster-manager 10.0.0.1:9000
```

### SGLang

```python
# adapter/sglang_connector.py — ~400 行
# 接口类似，适配 SGLang 的 BaseKVConnector
```

---

## 八、与现有项目的关系

### 借鉴来源

| 组件 | 主要参考 | 我们怎么做 |
|------|---------|----------|
| **磁盘读写** | LMCache LocalDiskBackend | 接口设计，Go 重写 |
| **GDS** | LMCache GdsBackend | CGo 包装 |
| **传输调度** | Mooncake StoreScheduler | gRPC + etcd，Go 实现 |
| **RDMA** | Mooncake Transfer Engine | CGo 包装 |
| **块管理** | vLLM KVCacheBlock | 引用计数 + 哈希 |
| **淘汰策略** | LMCache cache_policy | LRU/LFU/分层 |
| **文件格式** | ds4 .kv + LMCache .pt | 自定义格式 |

### 我们 vs Mooncake

```
                    Mooncake                      我们
──────────────────────────────────────────────────────────
定位            传输工具                     集群缓存操作系统
角色分配        手动配置                      自动（ClusterManager）
磁盘缓存         ❌ 无                        ✅ GDS + NVMe
存储节点         ❌ 必须有 GPU                 ✅ 纯磁盘也行
淘汰策略         ❌ 无                        ✅ LRU/LFU/分层
元数据          etcd（基本）                   etcd + RocksDB（完整）
vLLM 集成       ✅                           ✅
SGLang 集成     ✅                           ✅
RDMA            ✅                           ✅
```

---

## 九、实施路线图

### Phase 1：本地磁盘缓存 MVP（~6 周）

```
目标: 单机磁盘读写得通
├── Go 引擎核心
│   ├── CacheEngine 接口
│   ├── BlockManager（rocksdb）
│   └── LRU 淘汰策略
├── POSIX / io_uring 磁盘 I/O
├── Python vLLM connector（不含集群部分）
├── 集成测试: vLLM + 本地磁盘缓存
└── 验证: 长上下文推理不爆显存
```

### Phase 2：GDS 加速（~4 周）

```
目标: GPU 直读直写 NVMe
├── C GDS 封装（cufile）
├── Go CGo 绑定
├── 自动回退：GDS 不可用走 POSIX
└── 验证: 吞吐 3-5x vs CPU bounce buffer
```

### Phase 3：集群化（~8 周）

```
目标: 多节点共享磁盘缓存
├── Go ClusterManager
│   ├── etcd 注册/发现
│   ├── 角色分配引擎
│   ├── gRPC 接口
│   └── 故障转移
├── Connector 集成集群通信
├── RDMA 数据传输
├── SGLang adapter
└── 集成测试: 3 节点集群
```

### Phase 4：产品化（持续）

```
目标: 可交付的产品
├── Helm Chart（K8s 部署）
├── Prometheus / Grafana
├── 管理 Dashboard
├── 多租户
├── 文档 + 示例
└── 客户 POC
```

---

## 十、快速开始（设计稿）

```bash
# 1. 启动 ClusterManager（一个集群一个）
disk-cache cluster start \
    --etcd-endpoints 10.0.0.1:2379 \
    --listen :9000

# 2. 在每个节点上启动 vLLM（自动注册到集群）
vllm serve meta-llama/Llama-3.1-70B \
    --kv-connector disk-cache \
    --disk-cache-path /mnt/nvme/kv-cache \
    --disk-cache-size 2TB \
    --cluster-manager 10.0.0.1:9000

# 3. 部署纯 Storage 节点（没 GPU 也能加）
disk-cache storage start \
    --data-dir /mnt/nvme/disk-cache \
    --disk-size 30TB \
    --cluster-manager 10.0.0.1:9000

# 4. 集群状态
disk-cache cluster status

# 输出:
# Node         Role        GPU        Disk Used    Queue
# node-a       prefill     H100x8     45%          12
# node-b       decode      L40Sx8     32%          3
# node-c       storage     -          67%          -
```

---

## 十一、C/S 架构设计（v0.3）

### 11.1 为什么要 C/S

现有架构中，disk-cache engine 是"单机"的——每台机器跑一个独立进程，通过 HTTP API 被 Python connector 调用。没有统一的集群管理入口。

**需要 C/S 架构解决的问题：**

| 问题 | 现有方案 | C/S 方案 |
|------|---------|---------|
| 集群有多少节点在线？ | 不知道 | S端 统一点表 |
| 某台机器 vLLM 挂了？ | 得 SSH 上去看 | S端 心跳超时自动标记 offline |
| 加载新模型？ | 每台机器手动 exec | S端 一键下发指令 |
| KV cache 命中在哪台机器？ | 不知道 | S端 元数据中心返回精确位置 |
| 集群 cache 命中率？ | 没有 | S端 聚合统计 |

### 11.2 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         S端 (Server / 一个集群一个)                     │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │ 节点注册中心      │  │ 指令调度器       │  │ KV Cache 元数据中心   │ │
│  │ (Registry)      │  │ (Dispatcher)    │  │ (MetaCenter)         │ │
│  │                  │  │                 │  │                      │ │
│  │ • 心跳/保活      │  │ • exec 指令下发  │  │ • 现有 disk-cache    │ │
│  │ • 状态聚合       │  │ • 模型加载/卸载  │  │   Go 引擎 (Pebble)   │ │
│  │ • 集群拓扑       │  │ • 配置热更新     │  │                      │ │
│  │ • 超时/离线标记  │  │ • 广播执行      │  │ • hash → location   │ │
│  │ • 磁盘容量追踪   │  │ • 进度追踪      │  │ • 两种模式路由决策   │ │
│  └────────┬─────────┘  └───────┬─────────┘  └──────────┬───────────┘ │
│           │                    │                        │            │
│           └────────────────────┼────────────────────────┘            │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  rpcx Server (:9000)                                        │    │
│  │  ┌─ ClusterService ─┬─ CommandService ─┬─ QueryService ──┐ │    │
│  │  │ Register          │ FetchCommands    │ CacheLookup     │ │    │
│  │  │ Heartbeat         │ ReportResult     │ ClusterStatus   │ │    │
│  │  │ ReportCacheStatus │                  │ NodeDetail      │ │    │
│  │  └──────────────────┴─────────────────┴─────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  REST API (:8080) + Embedded Web UI                          │    │
│  │  GET  /api/v1/cluster/status  → Dashboard                     │    │
│  │  GET  /api/v1/nodes/:id      → Node Detail                    │    │
│  │  POST /api/v1/command        → Dispatch                       │    │
│  │  GET  /api/v1/cache/stats    → Cache Analytics                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
         ▲ rpcx TCP                    ▲ rpcx TCP
         │                             │
         ▼                             ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  C端 Agent node-1 │    │  C端 Agent node-2 │    │  C端 Agent node-3 │
│                    │    │                    │    │                    │
│  ┌──────────────┐ │    │  ┌──────────────┐ │    │  ┌──────────────┐ │
│  │ vLLM 进程管理  │ │    │  │ vLLM 进程管理  │ │    │  │ vLLM 进程管理  │ │
│  │ • exec 拉起   │ │    │  │ • exec 拉起   │ │    │  │ • exec 拉起   │ │
│  │ • 健康探活    │ │    │  │ • 健康探活    │ │    │  │ • 健康探活    │ │
│  │ • stdout 监控 │ │    │  │ • stdout 监控 │ │    │  │ • stdout 监控 │ │
│  └──────────────┘ │    │  └──────────────┘ │    │  └──────────────┘ │
│  ┌──────────────┐ │    │  ┌──────────────┐ │    │  ┌──────────────┐ │
│  │ 状态采集      │ │    │  │ 状态采集      │ │    │  │ 状态采集      │ │
│  │ • GPU 显存   │ │    │  │ • GPU 显存   │ │    │  │ • GPU 显存   │ │
│  │ • 磁盘(多盘) │ │    │  │ • 磁盘(多盘) │ │    │  │ • 磁盘(多盘) │ │
│  │ • vLLM 状态  │ │    │  │ • vLLM 状态  │ │    │  │ • vLLM 状态  │ │
│  │ • Cache 统计 │ │    │  │ • Cache 统计 │ │    │  │ • Cache 统计 │ │
│  └──────────────┘ │    │  └──────────────┘ │    │  └──────────────┘ │
│  ┌──────────────┐ │    │  ┌──────────────┐ │    │  ┌──────────────┐ │
│  │ rpcx Client  │ │    │  │ rpcx Client  │ │    │  │ rpcx Client  │ │
│  │ → S端 :9000  │ │    │  │ → S端 :9000  │ │    │  │ → S端 :9000  │ │
│  └──────────────┘ │    │  └──────────────┘ │    │  └──────────────┘ │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
  │ 本地 NVMe    │         │ 本地 NVMe    │         │ 共享存储     │
  │ /mnt/nvme0  │         │ /mnt/nvme0  │         │ /mnt/shared/ │
  │ /mnt/nvme1  │         │ /mnt/nvme1  │         │   kv-cache/  │
  └──────────────┘         └──────────────┘         └──────────────┘
```

### 11.3 通信协议（rpcx）

选用 rpcx 而非 gRPC 的理由：

| 对比维度 | gRPC | rpcx |
|---------|------|------|
| 服务定义 | protobuf 编译 | Go interface 原生 |
| 全 Go 项目 | 多一层代码生成 | 零代码生成 |
| 双向通信 | 需要 stream | rpcx 原生支持双向注册 |
| 部署体积 | 带 protobuf 运行时 | 纯 Go 零依赖 |
| 学习成本 | protobuf + 代码生成 | 就是写 Go 函数 |

**服务定义：**

```go
// ── S端 提供的服务（C端 调用）──

// ClusterService - 节点管理 + 心跳
type ClusterService interface {
    Register(ctx context.Context, info *NodeInfo, reply *RegisterReply) error
    Heartbeat(ctx context.Context, status *MachineStatus, reply *HeartbeatReply) error
}

// CommandService - 指令拉取 + 结果上报
type CommandService interface {
    FetchCommands(ctx context.Context, nodeID string, reply *[]*Command) error
    ReportResult(ctx context.Context, result *CmdResult, reply *OK) error
}

// QueryService - 缓存查询 + 集群查询
type QueryService interface {
    CacheLookup(ctx context.Context, hash uint64, reply *CacheLocation) error
    ClusterStatus(ctx context.Context, _ *Empty, reply *ClusterSummary) error
    NodeDetail(ctx context.Context, nodeID string, reply *NodeDetail) error
}

// ── C端 提供的服务（S端 主动推送，双向注册模式）──

// AgentService - S端 主动下发指令
type AgentService interface {
    ExecCommand(ctx context.Context, cmd *Command, reply *CmdResult) error
}
```

**心跳融合指令拉取：**

```
C端                      S端
 │                        │
 ├── Heartbeat(status) ──►│  ← 每 5s 发送
 │                        │  ├── 更新心跳时间戳
 │                        │  ├── 检查 pending commands
 │                        │  └── 打包 reply
 │◄── HeartbeatReply ─────┤
 │    ├── commands[]      │  ← 待执行指令（如果有）
 │    ├── role_change     │  ← 角色变更（如果有）
 │    └── config_update   │  ← 配置更新
 │                        │
 │  [有 pending cmd]      │
 │  └── ReportResult ────►│  ← 异步上报执行结果
 │                        │
 │  [长时间命令]           │
 │  └── Heartbeat(进度) ──►│  ← 下个心跳带 progress
```

### 11.4 核心数据结构

```go
// engine/pkg/cluster/types.go

// ── 磁盘信息（关键：多盘支持，ip + disk_path 路由）──

type DiskInfo struct {
    Path    string  // "/mnt/nvme0"
    TotalGB int64   // 总量
    UsedGB  int64   // 已用
    FreeGB  int64   // 剩余
}

type DiskUsage struct {
    Path   string
    FreeGB int64
    UsedGB int64
}

// ── 节点 ──

type NodeInfo struct {
    NodeID      string      // "node-1"
    Hostname    string
    IP          string      // 管理 IP
    RPCPort     int         // C端 rpcx 监听端口
    CacheMode   string      // "local_nvme" | "shared_pool"

    GPUType     string      // "H100"
    GPUMemMB    int64
    GPUCount    int

    Disks       []DiskInfo  // 多块盘
}

type NodeStatus string
const (
    NodeOnline  NodeStatus = "online"
    NodeOffline NodeStatus = "offline"
    NodeLoading NodeStatus = "loading"
    NodeError   NodeStatus = "error"
)

type MachineStatus struct {
    NodeID      string
    Timestamp   int64
    Seq         int64       // 单调递增，去重

    // 机器指标
    GPUUtil     float64     // 0.0-1.0
    GPUMemUsedMB int64
    MemUsedMB   int64
    CPULoad     float64

    // 磁盘
    Disks       []DiskUsage

    // vLLM 进程
    VLLMStatus  string     // "running"|"loading"|"stopped"|"error"
    ModelName   string
    QueueLen    int32
    LoadingPct  int32      // 0-100

    // KV Cache
    CacheBlocks int64
    CacheBytes  int64
    CacheHitRate float64
}

// ── 指令 ──

type CommandAction string
const (
    CmdStartVLLM   CommandAction = "start_vllm"
    CmdStopVLLM    CommandAction = "stop_vllm"
    CmdRestartVLLM CommandAction = "restart_vllm"
    CmdLoadModel   CommandAction = "load_model"
    CmdUnloadModel CommandAction = "unload_model"
    CmdUpdateConfig CommandAction = "update_config"
    CmdExecShell   CommandAction = "exec_shell"
)

type Command struct {
    CmdID     string
    Action    CommandAction
    Params    map[string]string
    Target    string           // 目标 node_id, "" = 广播
    CreatedAt int64
    Timeout   int              // 超时秒数
}

type CmdResult struct {
    CmdID    string
    NodeID   string
    Status   string   // "running"|"success"|"failed"|"timeout"
    Output   string
    Error    string
    Progress int32    // 0-100
}

// ── Cache 路由 ──

type CacheLocation struct {
    Hash     uint64
    Size     int64

    // 模式1 (local_nvme)：IP + 精确到盘
    NodeID   string
    IP       string
    DiskPath string  // "/mnt/nvme0"  ← 精确到哪块盘
    FilePath string  // "kv/blk_a1b2"

    // 模式2 (shared_pool)
    SharedPath string
}

// ── 集群查询 ──

type NodeSummary struct {
    NodeID      string
    IP          string
    Status      NodeStatus
    GPUUtil     float64
    GPUMemUsed  int64
    ModelName   string
    CacheBlocks int64
    HitRate     float64
    LastSeen    int64
    Disks       []DiskUsage
}

type ClusterSummary struct {
    Nodes       []NodeSummary
    CacheMode   string
    OnlineNodes int
    TotalNodes  int
    TotalBlocks int64
    TotalBytes  int64
    HitRate     float64
}

type NodeDetail struct {
    Info   *NodeInfo
    Status *MachineStatus
    Recent []*CmdResult  // 最近指令
}
```

### 11.5 两种 KV Cache 模式的路由决策

```
                       ┌──────────────────────────────┐
                       │  KV Cache 元数据中心            │
                       │  (现有 Go disk-cache engine)    │
                       │                                │
                       │  sentinel: hash → 存在?         │
                       │  block: hash → location         │
                       └──────────────┬───────────────┘
                                      │
                  ┌───────────────────┼────────────────────┐
                  │ cache_mode        │                    │
                  ▼                   ▼                    │
    ┌────────────────────────┐  ┌────────────────────┐    │
    │ 模式1: local_nvme       │  │ 模式2: shared_pool │    │
    │                        │  │                    │    │
    │ block 存在 node-X 的    │  │ block 存在共享存储   │    │
    │ /mnt/nvme1 上           │  │ /mnt/shared/...    │    │
    │                        │  │                    │    │
    │ 路由决策:               │  │ 路由决策:           │    │
    │ CacheLocation{         │  │ CacheLocation{     │    │
    │   NodeID: "node-3",   │  │   SharedPath: "...",│    │
    │   IP: "10.0.0.3",     │  │ }                  │    │
    │   DiskPath: "/nvme1", │  │                    │    │
    │ }                     │  │ → 任意节点可读       │    │
    │                        │  │                    │    │
    │ → 请求路由到 node-3    │  │ → 本地读共享存储     │    │
    │   的 /nvme1 读数据     │  │                     │    │
    └────────────────────────┘  └────────────────────┘    │
```

**关于 ip + disk_path 为什么要精确到盘：**

```
同一台机器的多盘场景:

┌─ node-3 (10.0.0.3) ─────────────────────────┐
│                                              │
│  /mnt/nvme0  (3.5TB NVMe)                   │
│    ├── 当前已用: 3.1TB  ██████████░░ 88%     │
│    └── 剩余: 0.4TB                           │
│                                              │
│  /mnt/nvme1  (3.5TB NVMe)                   │
│    ├── 当前已用: 0.5TB  █░░░░░░░░░ 14%       │
│    └── 剩余: 3.0TB             ← 新 block 写这里! │
│                                              │
│  /mnt/nvme2  (7.0TB NVMe)                   │
│    ├── 系统盘, 不参与 cache                    │
│    └── 剩余: 4.5TB                           │
└──────────────────────────────────────────────┘
```

S端 的 `disk_tracker` 模块负责：
- 维护每节点每盘 `(node_id, disk_path) → (total, used, free)`
- 新 cache block 写入时推荐剩余空间最大的盘
- 心跳上报中同步每盘最新状态

### 11.6 嵌入式 Web 管理后台

**技术选型：**

| 层 | 技术 | 理由 |
|----|------|------|
| 静态文件嵌入 | Go `embed` | 一个二进制部署 |
| 前端框架 | Alpine.js (CDN) | 15KB，声明式绑定 |
| 图表 | Chart.js (CDN) | 轻量图表 |
| 样式 | Pico CSS (CDN) | 语义化 HTML 自带美观 |
| REST 路由 | Go net/http 标准 mux | 简单够用 |

**页面规划：**

```
/                    → Dashboard（概览 + 节点列表）
/nodes.html          → 节点列表（表格 + 状态标签）
/node.html?id=xxx    → 单机详情（指标 + 磁盘 + 操作按钮）
/cache.html          → Cache 统计（命中率趋势 + block 分布）
/commands.html       → 指令管理（历史 + 下发表单）
```

**文件结构：**

```
pkg/server/web/
├── embed.go         //go:embed static/*
├── api.go           # REST API handlers
└── static/
    ├── index.html   # Dashboard
    ├── nodes.html   # 节点列表
    ├── node.html    # 单机详情
    ├── cache.html   # Cache 统计
    ├── commands.html# 指令管理
    ├── app.js       # Alpine.js 通用组件
    └── style.css    # 自定义样式
```

### 11.7 实施路线

| Phase | 内容 | 输出 |
|-------|------|------|
| **1** | 共享类型 + rpcx 服务接口 | `pkg/cluster/types.go`, `pkg/cluster/rpc_service.go` |
| **2** | S端 Server 核心 | `pkg/server/` (registry, dispatcher, rest api, web ui) |
| **3** | S端 入口命令 | `cmd/cluster-server/main.go` |
| **4** | C端 Agent | `pkg/agent/` + `cmd/c-agent/main.go` |
| **5** | 集成 disk-cache 作为元数据中心 | `pkg/server/router.go` + disk_tracker |
| **6** | Makefile + 端到端验证 | `make build-cs`, 本地双进程联调 |

---

## 十二、目录结构总览

```
engine/
├── cmd/
│   ├── disk-cache/          # [现有] 单机 HTTP 引擎
│   │   └── main.go
│   ├── cluster-server/      # [新增] S端
│   │   └── main.go
│   └── c-agent/             # [新增] C端
│       └── main.go
│
├── pkg/
│   ├── cache/               # [现有] 核心 cache 引擎
│   ├── metadata/            # [现有] Pebble 元数据存储
│   ├── eviction/            # [现有] LRU 淘汰
│   │
│   ├── cluster/             # [新增] C/S 共享
│   │   ├── types.go         #   数据结构
│   │   └── rpc_service.go   #   rpcx 服务接口
│   │
│   ├── server/              # [新增] S端
│   │   ├── server.go        #   主服务
│   │   ├── registry.go      #   节点注册
│   │   ├── dispatcher.go    #   指令调度
│   │   ├── router.go        #   Cache 路由
│   │   ├── disk_tracker.go  #   磁盘追踪
│   │   └── web/             #   REST API + Web UI
│   │       ├── embed.go
│   │       ├── api.go
│   │       └── static/*
│   │
│   └── agent/               # [新增] C端
│       ├── agent.go         #   主循环
│       ├── process.go       #   进程管理
│       ├── collector.go     #   状态采集
│       └── cache_proxy.go   #   本地 cache 代理
│
├── go.mod
└── go.sum
```

---



| 层 | 是否开源 | 理由 |
|---|---------|------|
| Python connector | ✅ 开源 | 吸引 vLLM/SGLang 社区 |
| Go 引擎核心 | ⏳ 待定 | 核心竞争壁垒 |
| C 存储原语 | ✅ 开源 | 硬件厂商需适配 |

---

> 文档版本: v0.2
> 最后更新: 2025-07-19
