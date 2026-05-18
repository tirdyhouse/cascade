# 集群磁盘 KV 缓存（DiskCache）

> 给 vLLM / SGLang 等推理框架加一个集群磁盘缓存层，
> 让 GPU 可以直接读写远端 NVMe SSD，降低 LLM 推理的显存需求。

## 项目结构

```
predict/
├── adapter/           # 推理框架适配器（Python）
│   ├── vllm/          # vLLM KVConnector 实现
│   └── sglang/        # SGLang KVConnector 实现
├── engine/            # Go 引擎核心
│   ├── cmd/           # 入口
│   └── pkg/           # 核心库
│       ├── cache/     # 缓存引擎
│       ├── cluster/   # 集群管理
│       ├── eviction/  # 淘汰策略
│       ├── metadata/  # 元数据管理
│       ├── storage/   # 存储后端
│       └── transport/ # 传输层
├── csrc/              # C 存储原语
│   ├── gds/           # GPUDirect Storage
│   ├── rdma/          # RDMA
│   └── iouring/       # io_uring
├── deploy/            # 部署
│   └── helm/          # Helm Chart
├── docs/              # 文档
└── others/            # 参考项目（已 gitignore）
```

## 快速开始

```bash
# 见 DESIGN.md
```

## 架构

详见 [DESIGN.md](./DESIGN.md)
