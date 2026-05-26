# 磁盘缓存命中设计方案

> Cascade 如何判断一个请求之前是否被缓存过，以及如何安全地读回 KV tensor 数据。

---

## 一、设计目标

1. 同一个 prompt 再次请求时能命中磁盘缓存，跳过重复的 prefill 计算
2. 不同长度的 prompt 可以共享前缀缓存（前 N 个 block 相同即可命中）
3. 命中检测在 Go 引擎中完成，通过 Pebble 元数据查询，避免文件系统 stat()
4. 匹配粒度对齐 vLLM 的 block_size（16 tokens）

---

## 二、核心概念：sentinel

**sentinel** 是一个标记，表示"某个前缀的 KV 数据已全部写入磁盘"，存储在 Go 引擎的 Pebble 数据库中。

```
sentinel{ SHA-256(tokens[0..N]) } = N

示例：
  写入 48 tokens → sentinel{hash_16} = 16, sentinel{hash_32} = 32, sentinel{hash_48} = 48
```

- **Key**：SHA-256 的 hex 字符串（前 32 字符）——对 prompt token IDs 的累积 hash
- **Value**：uint64 整数 —— 已被完整缓存的 token 数
- **存储**：Pebble，prefix `0x01` 区分于 block 元数据（prefix `0x00`）

---

## 三、写入流程

```
vLLM 推理过程
    │
    ├── save_kv_layer() × 24层: 将 KV tensor 写入 safetensors 文件
    │
    └── wait_for_save(): 所有层写完，调用 /record_batch
    
Go 引擎 RecordAll():
    1. 增量 SHA-256 扫描 token_ids 一次
       ├── block 0: hash_16 = SHA-256(tokens[0:16])
       ├── block 1: hash_32 = SHA-256(tokens[0:32])
       └── ...
       每个 block 边界保存哈希状态快照，追加 mm_hashes 后最终化
    
    2. Pebble 批量写入（已存在的跳过）
       sentinel{hash_16} = 16
       sentinel{hash_32} = 32
       ...
       sentinel{hash_N}   = N

复杂度: O(N) 哈希计算 + O(N/16) 次 Pebble 写入（毫秒级）
```

### 状态快照

SHA-256 的 `encoding.BinaryMarshaler/Unmarshaler` 接口支持中间状态保存：

```go
h := sha256.New()
for i := 0; i < numBlocks; i++ {
    h.Write(tokens[i*16 : (i+1)*16])   // 追加一个 block
    clone := cloneHash(h)               // 克隆当前状态
    clone.Write(mm_hashes)              // 追加多模态 hash
    hash_i = clone.Sum()                // 得到累积 hash
}
```

---

## 四、查询/命中流程

```
vLLM scheduler
    │
    └── get_num_new_matched_tokens() → POST /match {token_ids, block_size}
    
Go 引擎 Match():
    1. 增量 SHA-256 扫描 token_ids 一次（与写入相同的方式）
       得到 hashes = [hash_16, hash_32, hash_48, ...]
    
    2. 二分搜索
       lo, hi = 0, len(hashes)-1
       while lo <= hi:
           mid = (lo + hi) / 2
           if GetSentinel(hashes[mid]):  lo = mid + 1  (能找到，往大找)
           else:                          hi = mid - 1  (不能，往小找)
       找到的最大索引 hi 对应的 hash
    
    3. matched = (hi + 1) × block_size
       返回 {matched_tokens, prompt_hash}

复杂度: O(N) 哈希计算 + O(log N) 次 Pebble 查询（~13 次 for 128K tokens）
```

### 二分搜索正确性

`RecordAll` 保证了 sentinel 的**单调性**：如果 `hashes[k]` 存在，则 `hashes[0..k-1]` 也一定存在。

```
sentinel{hash_16} = 16  ← 一定存在
sentinel{hash_32} = 32  ← 如果存在，则 hash_16 也一定存在
sentinel{hash_48} = 48  ← 如果存在，则 hash_16 和 hash_32 也一定存在
```

因此二分搜索有效。

### 增量写入优化

多轮对话场景，每轮只写新增的 block sentinel：

```
Round 1 (160 tokens): 写 hash_16..hash_160 (10条)
Round 2 (320 tokens): 写 hash_176..hash_320 (10条，旧的跳过)
Round 3 (480 tokens): 写 hash_336..hash_480 (10条)
...
```

实现：`RecordAll` 中 `GetSentinel(hash)` 检查，已存在则 `continue`。

---

## 五、读回流程

```
命中后，vLLM 分配 GPU pages

start_load_kv():
    对于每层:
        根据 matched prompt_hash 定位 safetensors 文件
        从磁盘加载 KV tensor（bf16 原生格式）
        自动转换 dtype 匹配目标层
        通过 slot_mapping 注入 GPU 页表
    
    同时调用 /get API 通知 Go 引擎更新 BlocksRetrieved 统计
```

### 前缀匹配读回

```
缓存了 160 tokens (A)，请求 35 tokens (B，前缀相同):

  命中 matched = 32（对齐到 block）
  读回 safetensors 文件 → 取前 32 个 slot 注入 GPU
  vLLM 从 token 32 开始继续推理
```

---

## 六、粒度对齐

| 项目 | 粒度 | 说明 |
|------|------|------|
| vLLM block | 16 tokens | GPU 页表管理的基本单位 |
| 我们的 sentinel | 16 tokens | 对齐 vLLM，一个 sentinel 对应一个 block |
| hash 计算 | 累积 | hash_N = SHA-256(tokens[0..N]) |
| 文件存储 | 整 prompt | 一个 safetensors 文件存所有 token 的 KV |
| 读回 | 任意前缀 | 从文件中取前 matched 个 slot |

---

## 七、API 参考

| 端点 | 方法 | 用途 |
|------|------|------|
| `/match` | POST | `{token_ids, mm_hashes, block_size}` → `{matched_tokens, prompt_hash}` |
| `/record_batch` | POST | `{token_ids, mm_hashes, block_size}` → 记录所有子 block sentinel |
| `/record` | POST | `{prompt_hash, num_tokens}` → 记录单个 sentinel（兜底） |

---

## 八、复杂度

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|:----------:|:----------:|------|
| 写入 sentinel | O(tokens) | O(tokens/16) | 一次 SHA-256 扫描 + batch 写入 |
| 查询匹配 | O(tokens) + O(log blocks) | O(tokens/16) | 增量 hash + 二分搜索 Pebble |
| 读回 KV | O(tokens × hidden) | O(1) | safetensors 直接读 GPU |
| 多轮追加写入 | O(新增 tokens) | O(新增 blocks) | 已存在的 sentinel 跳过 |
