// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Cascade contributors.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Copy one layer-major source object into vLLM's Triton paged-KV layout.
//
// source_ptrs[layer] points at a contiguous [tokens, 2, hidden_bytes]
// tensor. destination_ptrs[layer] points at a contiguous
// [num_blocks, 2, block_size, hidden_bytes] tensor. One launch covers all
// layers, tokens, and K/V planes, avoiding a Python loop and one advanced
// indexing operation per layer.
__global__ void scatter_layer_major_kernel(
    const int64_t* __restrict__ source_ptrs,
    const int64_t* __restrict__ destination_ptrs,
    const int64_t* __restrict__ slot_mapping,
    int64_t num_tokens,
    int64_t hidden_bytes,
    int64_t block_size) {
  const int64_t token = blockIdx.x;
  const int64_t layer = blockIdx.y;
  const int64_t kv = blockIdx.z;
  if (token >= num_tokens) {
    return;
  }

  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }

  const auto* source = reinterpret_cast<const uint8_t*>(
      static_cast<uintptr_t>(source_ptrs[layer]));
  auto* destination = reinterpret_cast<uint8_t*>(
      static_cast<uintptr_t>(destination_ptrs[layer]));

  const int64_t source_offset = (token * 2 + kv) * hidden_bytes;
  const int64_t block = slot / block_size;
  const int64_t block_offset = slot - block * block_size;
  const int64_t destination_offset =
      ((block * 2 + kv) * block_size + block_offset) * hidden_bytes;

  // Qwen's per-token KV rows are naturally 16-byte aligned. Retain a byte
  // fallback for other model shapes so this operation remains exact rather
  // than imposing a hidden-size restriction.
  if ((hidden_bytes & 15) == 0) {
    const auto* source_words =
        reinterpret_cast<const uint4*>(source + source_offset);
    auto* destination_words =
        reinterpret_cast<uint4*>(destination + destination_offset);
    const int64_t word_count = hidden_bytes / 16;
    for (int64_t word = threadIdx.x; word < word_count;
         word += blockDim.x) {
      destination_words[word] = source_words[word];
    }
  } else {
    for (int64_t byte = threadIdx.x; byte < hidden_bytes;
         byte += blockDim.x) {
      destination[destination_offset + byte] = source[source_offset + byte];
    }
  }
}

void check_pointer_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64,
              name, " must have dtype int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
}

}  // namespace

void scatter_layer_major(
    const torch::Tensor& source_ptrs,
    const torch::Tensor& destination_ptrs,
    const torch::Tensor& slot_mapping,
    int64_t hidden_bytes,
    int64_t block_size) {
  check_pointer_tensor(source_ptrs, "source_ptrs");
  check_pointer_tensor(destination_ptrs, "destination_ptrs");
  check_pointer_tensor(slot_mapping, "slot_mapping");
  TORCH_CHECK(source_ptrs.numel() == destination_ptrs.numel(),
              "source and destination pointer counts must match");
  TORCH_CHECK(source_ptrs.numel() > 0, "at least one layer is required");
  TORCH_CHECK(slot_mapping.numel() > 0, "at least one token is required");
  TORCH_CHECK(hidden_bytes > 0, "hidden_bytes must be positive");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(source_ptrs.device() == destination_ptrs.device(),
              "pointer tensors must be on the same CUDA device");
  TORCH_CHECK(source_ptrs.device() == slot_mapping.device(),
              "slot_mapping must be on the pointer tensor CUDA device");

  const auto device_index = source_ptrs.get_device();
  c10::cuda::CUDAGuard device_guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();

  const dim3 grid(
      static_cast<unsigned int>(slot_mapping.numel()),
      static_cast<unsigned int>(source_ptrs.numel()),
      2);
  constexpr int threads = 128;
  scatter_layer_major_kernel<<<grid, threads, 0, stream>>>(
      source_ptrs.data_ptr<int64_t>(),
      destination_ptrs.data_ptr<int64_t>(),
      slot_mapping.data_ptr<int64_t>(),
      slot_mapping.numel(),
      hidden_bytes,
      block_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "scatter_layer_major",
      &scatter_layer_major,
      "Scatter all layers from a layer-major object into paged KV (CUDA)");
}
