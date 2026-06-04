# Future Integration: position-aware-embed

## Current State
Position-aware text embedding for command matching — 44% top-1 accuracy vs 0% for pure hash, sub-microsecond latency, zero ML dependencies. Hashes words with position weights (earlier words matter more), producing 64-dim vectors that enable fuzzy matching at hash speed.

## Integration Opportunities

### With ternary-esp32-firmware
The ESP32's lookup table is 81 entries for 4-trit inputs. With position-aware embeddings, the ESP32 becomes a fuzzy pattern matcher: incoming sensor patterns are embedded as 64-dim vectors, matched against stored patterns via cosine similarity. At 64 × 4 bytes = 256 bytes per pattern, 100 stored patterns = 25KB — well within the ESP32's 520KB SRAM. Same sub-microsecond latency, much richer response space.

### With lever-runner
position-aware-embed IS lever-runner's Gate 2 (vector similarity). The lever-runner-carapace Rust crate uses this embedding for fuzzy command matching. The WASM build (lever-runner-wasm) runs the same embeddings in the browser. The embedding is already deployed — it just needs ternary context.

### With construct-core Layer 0 (BareMetalConstruct)
At Layer 0, no allocations, no heap. Position-aware embeddings with fixed 64-dim vectors fit on the stack. The `query_lookup()` method becomes: hash the query into a 64-dim embedding, do cosine similarity against a fixed-size table, return the best match. O(1) lookup with fuzzy matching on bare metal.

## Dormant Ideas Now Unlockable
The embedding was standalone with no application context. Now the full ternary stack provides contexts at every hardware tier: ESP32 (bare metal), Jetson (GPU), Codespace (full compute). Each tier uses the same embedding algorithm but at different scales.

## Potential in Mature Systems
Every room uses position-aware embeddings for its "search" capability. When an agent enters a room and asks "what can this room do?", the query is embedded and matched against the room's skill descriptions. When a cell needs to find similar neighbors, their state vectors are embedded and compared. The embedding is the universal similarity primitive.

## Cross-Pollination Ideas
- **ptx-bench**: GPU embedding benchmarks validate vectorized embedding throughput
- **torch-vector-search**: PyTorch GPU-accelerated search complements position-aware embeddings for large-scale queries
- **open-vectors/weaviate**: Position-aware embeddings stored in Weaviate for fleet-wide skill search

## Dependencies for Next Steps
- ESP32 firmware integration with fixed-pattern matching
- Integration with construct-core Layer 0 stack-only query path
- Weaviate collection for fleet-wide embedding search
