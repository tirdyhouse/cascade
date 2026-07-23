package cache

import (
	"fmt"
	"testing"
)

func BenchmarkV2MatchResolve(b *testing.B) {
	for _, n := range []int{16, 64} {
		for _, shards := range []int{1, 4} {
			n, shards := n, shards
			b.Run(fmt.Sprintf("match/chunks=%d/shards=%d", n, shards), func(b *testing.B) {
				root := b.TempDir()
				eng, err := New(Config{
					CachePath:      root + "/cache",
					MetadataPath:   root + "/meta",
					MaxSizeBytes:   1 << 30,
					EvictionPolicy: "lru",
				})
				if err != nil {
					b.Fatal(err)
				}
				defer eng.Close()

				objects := make([]ChunkObject, 0, n*shards)
				candidates := make([]ChunkCandidate, n)
				requiredShards := make([]string, shards)
				for shard := range shards {
					requiredShards[shard] = fmt.Sprintf("s%d", shard)
				}
				for i := range n {
					key := fmt.Sprintf("key-%04d", i)
					candidates[i] = ChunkCandidate{Key: key, EndTokens: (i + 1) * 256}
					for shard, shardName := range requiredShards {
						objects = append(objects, ChunkObject{
							Namespace:   "ns",
							Key:         key,
							Shard:       shardName,
							FilePath:    fmt.Sprintf("p/%s-%d", key, shard),
							Index:       i,
							StartTokens: i * 256,
							EndTokens:   (i + 1) * 256,
							Size:        1024,
						})
					}
				}
				if err := eng.CommitChunks(objects); err != nil {
					b.Fatal(err)
				}

				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if _, err := eng.MatchChunks("ns", candidates, requiredShards); err != nil {
						b.Fatal(err)
					}
				}
			})
		}

		b.Run(fmt.Sprintf("resolve/chunks=%d", n), func(b *testing.B) {
			root := b.TempDir()
			eng, err := New(Config{
				CachePath:      root + "/cache",
				MetadataPath:   root + "/meta",
				MaxSizeBytes:   1 << 30,
				EvictionPolicy: "lru",
			})
			if err != nil {
				b.Fatal(err)
			}
			defer eng.Close()

			objects := make([]ChunkObject, 0, n)
			keys := make([]string, n)
			for i := range n {
				key := fmt.Sprintf("key-%04d", i)
				keys[i] = key
				objects = append(objects, ChunkObject{
					Namespace:   "ns",
					Key:         key,
					Shard:       "s0",
					FilePath:    fmt.Sprintf("p/%s", key),
					Index:       i,
					StartTokens: i * 256,
					EndTokens:   (i + 1) * 256,
					Size:        1024,
				})
			}
			if err := eng.CommitChunks(objects); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := eng.ResolveChunks("ns", keys, "s0"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
