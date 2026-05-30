from adapter.vllm.connector_common import DiskCacheConnectorCommonMixin, DiskCacheMeta, logger
from adapter.vllm.hashing import align_to_block_size

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1


class DiskCacheConnector(DiskCacheConnectorCommonMixin, KVConnectorBase_V1):
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        meta = self._get_connector_metadata()
        if not self._is_disk_cache_meta(meta):
            return
        for req in meta.requests:
            if not req.is_store:
                continue
            self._save_request_kv(req, layer_name, kv_layer, attn_metadata)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        token_ids = request.prompt_token_ids or []
        if len(token_ids) < 2:
            return 0, False
        num_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)
        if num_to_check <= num_computed_tokens:
            return 0, False
        mm_hashes = [f.identifier for f in request.mm_features]
        result = self._go_match(token_ids, mm_hashes)
        if result and result.get("matched_tokens", 0) > num_computed_tokens:
            matched = result["matched_tokens"]
            # Leave at least 1 block for vLLM to compute, otherwise
            # scheduler assert num_new_tokens > 0 fails.
            total_computed = matched
            if total_computed >= num_to_check:
                total_computed = num_to_check - self._block_size
            if total_computed <= num_computed_tokens:
                return 0, False
            ext_return = total_computed - num_computed_tokens
            logger.info("Disk cache HIT for request %s (%d tokens)", request.request_id, total_computed)
            return ext_return, False
        return 0, False


KVConnectorClass = DiskCacheConnector
