# position-aware-embed

Position-aware text embedding for command matching — 44% top-1 accuracy vs 0% for pure hash, sub-microsecond latency, zero ML dependencies.

## How It Works

1. Split text into words
2. Hash each word with its position: `blake2b(format!("{}:{}", i, word))`
3. Weight by position: `weight = 1.0 / (1 + i * 0.5)` (earlier words matter more)
4. Sum weighted vectors
5. L2-normalize

This gives semantically similar text higher cosine similarity than pure hashing, while keeping latency under 1µs.

## Usage

```rust
use position_aware_embed::{position_aware_embed, cosine_similarity, VectorIndex};

// Direct embedding
let v1 = position_aware_embed("check disk usage", 64);
let v2 = position_aware_embed("show disk usage", 64);
let sim = cosine_similarity(&v1, &v2);

// Vector index for command matching
let mut idx = VectorIndex::new(64);
idx.add("check disk usage", "df -h");
idx.add("check memory usage", "free -h");
idx.add("list docker containers", "docker ps");

let results = idx.search("check disk", 2);
```

## Benchmarks

```bash
cargo bench
```

## License

MIT
