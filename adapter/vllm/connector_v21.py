from adapter.vllm.connector_common import DiskCacheConnectorCommonMixin

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1


class DiskCacheConnector(DiskCacheConnectorCommonMixin, KVConnectorBase_V1):
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self._connected:
            return
        meta = self._get_connector_metadata()
        if not self._is_disk_cache_meta(meta):
            return
        for req in meta.requests:
            if not req.is_store:
                continue
            self._save_request_kv(req, layer_name, kv_layer, attn_metadata)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self._get_num_new_matched_tokens(request, num_computed_tokens)


KVConnectorClass = DiskCacheConnector
