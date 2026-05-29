"""Shared tensor helpers for vLLM disk-cache connectors."""

from __future__ import annotations


def inject_kv_into_layer(dst, src, slot_mapping, attn_metadata, block_size):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    if isinstance(attn_metadata, TritonAttentionMetadata):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        dst[block_idxs, :, offsets] = src
    else:
        num_pages = dst.shape[1]
        page_size = dst.shape[2]
        dst_flat = dst.reshape(2, num_pages * page_size, -1)
        dst_flat[:, slot_mapping, ...] = src


def extract_kv_from_layer(layer, slot_mapping, attn_metadata, block_size):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    if isinstance(attn_metadata, TritonAttentionMetadata):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        return layer[block_idxs, :, offsets]
    num_pages, page_size = layer.shape[1], layer.shape[2]
    return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]
