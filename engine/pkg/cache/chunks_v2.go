package cache

import (
	"fmt"
	"strconv"
	"strings"

	"predict/engine/pkg/metadata"
)

// v2PolicyKey builds a reversible policy key for a v2 chunk entry.
// Format: "v2:NNNN:namespace:NNNN:key:NNNN:shard"
// where NNNN is the 4-hex-digit length of the following component.
func v2PolicyKey(namespace, key, shard string) string {
	return fmt.Sprintf("v2:%04x:%s:%04x:%s:%04x:%s",
		len(namespace), namespace,
		len(key), key,
		len(shard), shard)
}

// v2PolicyIdentity is the comparable lookup key for an in-memory v2 entry.
type v2PolicyIdentity struct {
	namespace string
	key       string
	shard     string
}

// v2EntryInfo stores reverse-mapping data for a v2 LRU policy entry.
type v2EntryInfo struct {
	policyKey   string
	filePath    string
	size        int64
	index       int
	startTokens int
	endTokens   int
}

func v2PolicyIdentityFor(namespace, key, shard string) v2PolicyIdentity {
	return v2PolicyIdentity{namespace: namespace, key: key, shard: shard}
}

// parseV2PolicyKey decodes a policy key created by v2PolicyKey.
func parseV2PolicyKey(pk string) (namespace, key, shard string, ok bool) {
	if !strings.HasPrefix(pk, "v2:") {
		return "", "", "", false
	}

	remaining := pk[len("v2:"):]
	readComponent := func(input string, final bool) (value, rest string, ok bool) {
		separator := strings.IndexByte(input, ':')
		if separator <= 0 {
			return "", "", false
		}
		length, err := strconv.ParseUint(input[:separator], 16, 16)
		if err != nil {
			return "", "", false
		}
		start := separator + 1
		end := start + int(length)
		if end > len(input) {
			return "", "", false
		}
		if final {
			if end != len(input) {
				return "", "", false
			}
			return input[start:end], "", true
		}
		if end >= len(input) || input[end] != ':' {
			return "", "", false
		}
		return input[start:end], input[end+1:], true
	}

	namespace, remaining, ok = readComponent(remaining, false)
	if !ok {
		return "", "", "", false
	}
	key, remaining, ok = readComponent(remaining, false)
	if !ok {
		return "", "", "", false
	}
	shard, _, ok = readComponent(remaining, true)
	if !ok {
		return "", "", "", false
	}
	return namespace, key, shard, true
}

// validateChunkObjects validates a batch of ChunkObject for CommitChunks.
func validateChunkObjects(objects []ChunkObject) error {
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidArgument, fmt.Sprintf(format, args...))
	}
	if len(objects) == 0 {
		return invalid("empty objects")
	}
	seen := make(map[v2PolicyIdentity]struct{}, len(objects))
	namespace := objects[0].Namespace
	for i, obj := range objects {
		if obj.Namespace == "" {
			return invalid("objects[%d]: empty namespace", i)
		}
		if obj.Namespace != namespace {
			return invalid("objects[%d]: mixed namespace %q", i, obj.Namespace)
		}
		if obj.Key == "" {
			return invalid("objects[%d]: empty key", i)
		}
		if obj.Shard == "" {
			return invalid("objects[%d]: empty shard", i)
		}
		if len(obj.Namespace) > 65535 {
			return invalid("objects[%d]: namespace too long (%d bytes)", i, len(obj.Namespace))
		}
		if len(obj.Key) > 65535 {
			return invalid("objects[%d]: key too long (%d bytes)", i, len(obj.Key))
		}
		if len(obj.Shard) > 65535 {
			return invalid("objects[%d]: shard too long (%d bytes)", i, len(obj.Shard))
		}
		if _, err := rootedCachePath("", obj.FilePath); err != nil {
			return invalid("objects[%d]: %v", i, err)
		}
		if obj.Index < 0 {
			return invalid("objects[%d]: negative index (%d)", i, obj.Index)
		}
		if obj.StartTokens < 0 {
			return invalid("objects[%d]: negative start_tokens (%d)", i, obj.StartTokens)
		}
		if obj.EndTokens <= obj.StartTokens {
			return invalid("objects[%d]: end_tokens (%d) <= start_tokens (%d)", i, obj.EndTokens, obj.StartTokens)
		}
		if obj.Size <= 0 {
			return invalid("objects[%d]: non-positive size (%d)", i, obj.Size)
		}
		identity := v2PolicyIdentityFor(obj.Namespace, obj.Key, obj.Shard)
		if _, ok := seen[identity]; ok {
			return invalid("objects[%d]: duplicate namespace/key/shard", i)
		}
		seen[identity] = struct{}{}
	}
	return nil
}

// ── Engine v2 method implementations ──

// CommitChunks atomically commits a batch of v2 chunk objects.
func (e *diskEngine) CommitChunks(objects []ChunkObject) error {
	if err := validateChunkObjects(objects); err != nil {
		return err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	metas := make([]metadata.ChunkObjectMeta, len(objects))
	var totalNewSize int64
	for i, obj := range objects {
		metas[i] = metadata.ChunkObjectMeta{
			Namespace:   obj.Namespace,
			Key:         obj.Key,
			Shard:       obj.Shard,
			FilePath:    obj.FilePath,
			Index:       obj.Index,
			StartTokens: obj.StartTokens,
			EndTokens:   obj.EndTokens,
			Size:        obj.Size,
		}

		identity := v2PolicyIdentityFor(obj.Namespace, obj.Key, obj.Shard)
		if existing, ok := e.v2PolicyMap[identity]; ok {
			if existing.filePath != obj.FilePath ||
				existing.index != obj.Index ||
				existing.startTokens != obj.StartTokens ||
				existing.endTokens != obj.EndTokens ||
				existing.size != obj.Size {
				return fmt.Errorf("%w: conflict for namespace=%q key=%q shard=%q", ErrInvalidArgument, obj.Namespace, obj.Key, obj.Shard)
			}
			continue
		}
		if !e.cfg.DisableEviction && obj.Size > e.cfg.MaxSizeBytes-totalNewSize {
			return fmt.Errorf("%w: batch size exceeds cache capacity %d", ErrInvalidArgument, e.cfg.MaxSizeBytes)
		}
		totalNewSize += obj.Size
	}

	if err := e.lockedMakeRoom(totalNewSize); err != nil {
		return err
	}

	newCount, err := e.v2.CommitChunkObjects(metas)
	if err != nil {
		return fmt.Errorf("v2 commit: %w", err)
	}

	for _, obj := range objects {
		pk := v2PolicyKey(obj.Namespace, obj.Key, obj.Shard)
		e.pol.Record(pk, obj.Size)
		e.v2PolicyMap[v2PolicyIdentityFor(obj.Namespace, obj.Key, obj.Shard)] = &v2EntryInfo{
			policyKey:   pk,
			filePath:    obj.FilePath,
			size:        obj.Size,
			index:       obj.Index,
			startTokens: obj.StartTokens,
			endTokens:   obj.EndTokens,
		}
	}

	e.chunksStored.Add(int64(newCount))
	return nil
}

// MatchChunks performs linear sequential match over candidates.
// For each candidate in order, verifies ALL required shards exist with matching EndTokens.
// Stops at the first candidate that is missing any required shard or has mismatched EndTokens.
// On match, refreshes LRU recency for matched entries.
func (e *diskEngine) MatchChunks(namespace string, candidates []ChunkCandidate, requiredShards []string) (ChunkMatchResult, error) {
	e.matchRequests.Add(1)
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidArgument, fmt.Sprintf(format, args...))
	}

	if namespace == "" {
		return ChunkMatchResult{}, invalid("empty namespace")
	}
	if len(candidates) == 0 {
		return ChunkMatchResult{}, nil
	}
	if len(requiredShards) == 0 {
		return ChunkMatchResult{}, invalid("empty required_shards")
	}

	// Validate and deduplicate requiredShards
	seen := make(map[string]bool, len(requiredShards))
	uniqueShards := make([]string, 0, len(requiredShards))
	for _, s := range requiredShards {
		if s == "" {
			return ChunkMatchResult{}, invalid("empty required_shard entry")
		}
		if !seen[s] {
			seen[s] = true
			uniqueShards = append(uniqueShards, s)
		}
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Pre-validate all candidates: key non-empty and strictly increasing end_tokens
	prevEnd := 0
	for i, cand := range candidates {
		if cand.Key == "" {
			return ChunkMatchResult{}, invalid("candidate[%d]: empty key", i)
		}
		if cand.EndTokens <= prevEnd {
			return ChunkMatchResult{}, invalid("candidate[%d]: end_tokens %d not strictly increasing after %d", i, cand.EndTokens, prevEnd)
		}
		prevEnd = cand.EndTokens
	}

	prevEnd = 0
	var matchedKeys []string
	matchedChunks := 0
	matchedTokens := 0

	for _, cand := range candidates {
		// Verify ALL required shards exist with matching EndTokens. The policy
		// map is rebuilt from and updated with the same metadata store while
		// holding e.mu, so it is the in-memory read index for this hot path.
		allShardsMatch := true
		for _, shard := range uniqueShards {
			identity := v2PolicyIdentityFor(namespace, cand.Key, shard)
			info, ok := e.v2PolicyMap[identity]
			if !ok || info.endTokens != cand.EndTokens {
				allShardsMatch = false
				break
			}
		}

		if !allShardsMatch {
			break
		}

		// Refresh LRU recency for all shards of this matched candidate
		for _, shard := range uniqueShards {
			identity := v2PolicyIdentityFor(namespace, cand.Key, shard)
			if info, ok := e.v2PolicyMap[identity]; ok {
				e.pol.Record(info.policyKey, info.size)
			}
		}

		if matchedKeys == nil {
			matchedKeys = make([]string, 0, len(candidates))
		}
		matchedChunks++
		matchedTokens = cand.EndTokens
		matchedKeys = append(matchedKeys, cand.Key)
	}

	if matchedChunks > 0 {
		e.matchHits.Add(1)
		e.matchedTokens.Add(int64(matchedTokens))
	}

	return ChunkMatchResult{
		MatchedChunks: matchedChunks,
		MatchedTokens: matchedTokens,
		MatchedKeys:   matchedKeys,
	}, nil
}

// ResolveChunks resolves chunk keys in order, returning the corresponding ChunkObject
// entries for the given shard. Returns an error if any key is missing.
// On success, refreshes LRU recency for resolved entries.
func (e *diskEngine) ResolveChunks(namespace string, keys []string, shard string) ([]ChunkObject, error) {
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidArgument, fmt.Sprintf(format, args...))
	}
	if namespace == "" {
		return nil, invalid("empty namespace")
	}
	if len(keys) == 0 {
		return nil, invalid("empty keys")
	}
	if shard == "" {
		return nil, invalid("empty shard")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	result := make([]ChunkObject, 0, len(keys))
	for _, key := range keys {
		if key == "" {
			return nil, invalid("empty key in keys list")
		}
		identity := v2PolicyIdentityFor(namespace, key, shard)
		info, ok := e.v2PolicyMap[identity]
		if !ok {
			return nil, fmt.Errorf("chunk not found: namespace=%q key=%q shard=%q", namespace, key, shard)
		}
		result = append(result, ChunkObject{
			Namespace:   namespace,
			Key:         key,
			Shard:       shard,
			FilePath:    info.filePath,
			Index:       info.index,
			StartTokens: info.startTokens,
			EndTokens:   info.endTokens,
			Size:        info.size,
		})

		// Refresh LRU recency.
		e.pol.Record(info.policyKey, info.size)
	}
	return result, nil
}

// InvalidateChunk removes one corrupt or missing v2 object from metadata and
// eviction accounting. The connector owns physical cleanup because it is the
// component that has conclusively validated the local object as bad.
func (e *diskEngine) InvalidateChunk(namespace, key, shard string) error {
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidArgument, fmt.Sprintf(format, args...))
	}
	if namespace == "" {
		return invalid("empty namespace")
	}
	if key == "" {
		return invalid("empty key")
	}
	if shard == "" {
		return invalid("empty shard")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	identity := v2PolicyIdentityFor(namespace, key, shard)
	info, ok := e.v2PolicyMap[identity]
	policyKey := ""
	if ok {
		policyKey = info.policyKey
	} else {
		meta, err := e.v2.GetV2Chunk(namespace, key, shard)
		if err != nil {
			return fmt.Errorf("read v2 chunk before invalidation: %w", err)
		}
		if meta == nil {
			return nil
		}
		policyKey = v2PolicyKey(namespace, key, shard)
	}

	if err := e.v2.DeleteV2Chunk(namespace, key, shard); err != nil {
		return fmt.Errorf("delete v2 chunk metadata: %w", err)
	}
	delete(e.v2PolicyMap, identity)
	e.pol.Remove(policyKey)
	return nil
}

// lockedEvictV2Entry removes a v2 entry from metadata and policy during eviction.
// Returns the file path for the caller to delete the actual file.
// Must be called with e.mu held.
// Only removes from policy map if metadata delete succeeds.
func (e *diskEngine) lockedDeleteV2Entry(pk string) (BlockMeta, bool) {
	namespace, key, shard, ok := parseV2PolicyKey(pk)
	if !ok {
		e.evictErrors.Add(1)
		return BlockMeta{}, false
	}
	identity := v2PolicyIdentityFor(namespace, key, shard)
	info, ok := e.v2PolicyMap[identity]
	if !ok {
		return BlockMeta{}, false
	}
	fullPath, err := rootedCachePath(e.cfg.CachePath, info.filePath)
	if err != nil {
		e.evictErrors.Add(1)
		return BlockMeta{}, false
	}
	if err := e.v2.DeleteV2Chunk(namespace, key, shard); err != nil {
		e.evictErrors.Add(1)
		return BlockMeta{}, false
	}
	delete(e.v2PolicyMap, identity)
	e.blocksEvicted.Add(1)
	return BlockMeta{FilePath: fullPath, Size: info.size}, true
}

// RecordRetrievedV2 records successful v2 chunk retrievals.
func (e *diskEngine) RecordRetrievedV2(count int) {
	if count <= 0 {
		return
	}
	e.chunksRetrieved.Add(int64(count))
}
