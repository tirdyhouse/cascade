"""Shared tensor helpers for vLLM disk-cache connectors.

Canonical layout convention
---------------------------
All extracted KV tensors use **token-first** layout: ``[T, 2, ...]``
where ``T`` is the number of selected tokens and ``2`` is the K/V
dimension, and the tensor is contiguous.
"""

from __future__ import annotations


def inject_kv_into_layer(dst, src, slot_mapping, attn_metadata, block_size):
    """Inject extracted KV *src* (``[T, 2, …]``) back into *dst*.

    For the Triton backend the assignment is direct.
    For the non-Triton backend *src* is moved from ``[T, 2, …]``
    to ``[2, T, …]`` before assignment.
    """
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    if isinstance(attn_metadata, TritonAttentionMetadata):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        dst[block_idxs, :, offsets] = src
    else:
        # src is [T, 2, …]; dst_flat expects [2, T, …]
        src_2T = src.movedim(0, 1).contiguous()
        num_pages = dst.shape[1]
        page_size = dst.shape[2]
        dst_flat = dst.reshape(
            2, num_pages * page_size, *dst.shape[3:]
        )
        dst_flat[:, slot_mapping, ...] = src_2T


def extract_kv_from_layer(layer, slot_mapping, attn_metadata, block_size):
    """Extract KV from *layer* and return ``[T, 2, …]`` contiguous.

    For the Triton backend the extracted slice is made contiguous
    after advanced indexing.
    For the non-Triton backend the layer is reshaped from ``[2, T, …]``
    to ``[T, 2, …]`` via ``movedim``.
    """
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    if isinstance(attn_metadata, TritonAttentionMetadata):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        result = layer[block_idxs, :, offsets]
        # Advanced indexing breaks contiguity; restore it.
        return result.contiguous()

    # Non-Triton: layer is [2, num_pages, page_size, …]
    num_pages, page_size = layer.shape[1], layer.shape[2]
    # flat shape = [2, T, …]  where T = num_pages * page_size
    flat_2T = layer.reshape(
        2, num_pages * page_size, *layer.shape[3:]
    )
    # Select tokens → [2, len(slot_mapping), …], then move K/V dim
    selected = flat_2T[:, slot_mapping, ...]
    return selected.movedim(0, 1).contiguous()
