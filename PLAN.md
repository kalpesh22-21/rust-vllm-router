# Cache-Aware Policy: Token & KV-Aware Scoring

## Motivation

The current cache-aware policy has two limitations:

1. **No tokenization**: the radix tree stores raw characters, not token IDs. A character-level prefix match does not map 1:1 to real KV cache slots, and mid-token node splits can produce false "hits".
2. **Load balancing ignores tokens and KV pressure**: only `running_requests` (a request count) is used. Two workers with the same request count can have very different memory pressure depending on sequence lengths and KV cache fill.

## Target Scoring Formula

```
# Hard exclusion
if kv_utilization(w) >= kv_high_watermark: skip worker

# Load penalty
token_pressure(w) = (running_tokens(w) / token_capacity)^2
kv_pressure(w)    = kv_utilization(w)^3
load_penalty(w)   = α * token_pressure(w) + β * kv_pressure(w)

# Cache value weighted by prompt length
cache_value(w) = cache_match(w) * ln(1 + input_tokens)

# Final score
score(w) = cache_value(w) - λ * load_penalty(w)

# Route to
argmax_w(score(w))
```

Fallback: if all workers exceed `kv_high_watermark`, route to the worker with the lowest KV utilization rather than returning an error.

## Request Flow (After Implementation)

```
Incoming request
    │
    ▼
1.  Extract model + prompt text
2.  POST /tokenize to a randomly selected healthy worker for that model
    → receive token_ids: Vec<u32> + input_tokens: usize
    │
    ▼
3.  For each healthy candidate worker:
      a. Read kv_utilization from short-TTL async cache
         - If cache is stale: spawn async refresh (fire-and-forget), use stale value now
      b. cache_match  = tree.prefix_match(token_ids)          // token-based tree
      c. cache_value  = cache_match * ln(1.0 + input_tokens)
      d. token_pressure = (running_tokens / token_capacity)^2
      e. kv_pressure    = kv_utilization^3
      f. load_penalty   = α * token_pressure + β * kv_pressure
      g. score          = cache_value - λ * load_penalty
      h. Skip if kv_utilization >= kv_high_watermark
    │
    ▼
4.  Route to argmax(score)
5.  Forward request with prompt_token_ids field (skip re-tokenization on pod)
6.  On completion: decrement running_tokens by the request's token count
7.  Insert token_ids into the radix tree for this worker
```

---

## Implementation Phases

### Phase 1 — Worker State Extensions

**Files:** `src/core/worker.rs`

Add to `BasicWorker`:

```rust
running_tokens: Arc<AtomicUsize>,
kv_cache_utilization: Arc<RwLock<(f32, Instant)>>,
```

Extend the `Worker` trait:

```rust
fn running_tokens(&self) -> usize;
fn kv_cache_utilization(&self) -> f32;
fn increment_load_with_tokens(&self, tokens: usize);
fn decrement_load_with_tokens(&self, tokens: usize);
fn set_kv_cache_utilization(&self, value: f32);
```

No behavior change — existing `increment_load` / `decrement_load` remain for callers that don't yet have token counts.

---

### Phase 2 — KV Cache Async Refresh (Demand-Driven)

**Files:** `src/core/worker.rs` (or a small `src/core/kv_refresh.rs`)

No background thread. The refresh is triggered lazily on the hot path:

```rust
fn kv_cache_utilization(&self) -> f32 {
    let (value, refreshed_at) = *self.kv_cache_utilization.read();
    if refreshed_at.elapsed() > kv_ttl {
        let worker = self.clone();
        tokio::spawn(async move {
            if let Ok(v) = fetch_kv_utilization(worker.url()).await {
                worker.set_kv_cache_utilization(v);
            }
        });
    }
    value  // always returns cached value immediately, never blocks
}
```

`fetch_kv_utilization` issues a GET to `{worker_url}/metrics`, parses the Prometheus text format, and extracts `vllm:gpu_cache_usage_perc`.

TTL is configurable (`kv_ttl_ms`, default 150ms). Under high RPS the cached value is used for the vast majority of requests; refreshes are infrequent relative to request rate.

---

### Phase 3 — `/tokenize` Integration

**Files:** `src/policies/cache_aware.rs`, `src/policies/mod.rs`

Before scoring, call the vLLM `/tokenize` endpoint on a **randomly selected healthy worker** for the model (random selection distributes load across workers):

```
POST {worker_url}/tokenize
{ "model": "...", "prompt": "..." }

Response:
{ "tokens": [1, 2938, 29879, ...], "count": 42 }
```

No caching of tokenization results — intra-cluster latency (~0.5–1ms) is acceptable and avoids cache complexity. The result (`Vec<u32>`, `usize`) is passed into the scoring step.

The inference request is forwarded with `prompt_token_ids` instead of the original text, so the inference worker does not re-tokenize.

---

### Phase 4 — Token-Based Radix Tree (same PR as Phase 3)

**Files:** `src/tree.rs`

Switch the tree from character-based to token-ID-based:

| Before | After |
|---|---|
| `Node.text: RwLock<NodeText>` (String) | `Node.text: RwLock<Vec<u32>>` |
| `prefix_match_with_counts(text: &str)` | `prefix_match_with_counts(tokens: &[u32])` |
| `insert(text: &str, tenant)` | `insert(tokens: &[u32], tenant)` |
| LRU evicts by `tenant_char_count` | LRU evicts by `tenant_token_count` |

**Why this is faster:**
- `u32` slice comparison is SIMD-vectorizable; UTF-8 string comparison is not.
- Tree depth drops ~4x for typical English prompts (1 token ≈ 4 chars).
- No mid-token node splits — every node boundary is a real KV cache boundary.

`tenant_char_count` in the `Tree` struct is renamed to `tenant_token_count`. Eviction logic is otherwise unchanged — it still evicts the LRU tenant when `tenant_token_count > max_tree_size` (now measured in tokens, not chars).

---

### Phase 5 — Config Extensions

**Files:** `src/config/types.rs`, `src/policies/mod.rs`

Add to `CacheAwareConfig`:

```rust
// Scoring weights
pub alpha: f32,               // token_pressure weight       (default: 1.0)
pub beta: f32,                // kv_pressure weight          (default: 1.0)
pub lambda: f32,              // load_penalty scale          (default: 1.0)

// Thresholds
pub kv_high_watermark: f32,   // hard exclusion threshold    (default: 0.90)
pub token_capacity: usize,    // max tokens per worker       (default: 32768)

// KV cache refresh
pub kv_ttl_ms: u64,           // short-TTL cache TTL         (default: 150)
```

Deprecate (keep with warnings for one release, then remove):
- `balance_abs_threshold`
- `balance_rel_threshold`

These are replaced by the unified score. Setting `lambda = 0` reproduces load-blind cache routing; setting all weights to zero reproduces round-robin.

---

### Phase 6 — New Scoring in `cache_aware.rs` + Token Count in Completion Callback

**Files:** `src/policies/cache_aware.rs`, `src/policies/mod.rs`

Replace the current two-mode logic (`if imbalanced → min_load else → cache_match`) with the unified formula. Core routing function becomes:

```rust
fn score_worker(
    worker: &dyn Worker,
    cache_match: f32,
    input_tokens: usize,
    config: &CacheAwareConfig,
) -> Option<f32> {
    let kv_util = worker.kv_cache_utilization();

    if kv_util >= config.kv_high_watermark {
        return None;  // hard exclusion
    }

    let token_pressure = {
        let r = worker.running_tokens() as f32 / config.token_capacity as f32;
        r * r
    };
    let kv_pressure = kv_util * kv_util * kv_util;
    let load_penalty = config.alpha * token_pressure + config.beta * kv_pressure;

    let cache_value = cache_match * (1.0 + input_tokens as f32).ln();

    Some(cache_value - config.lambda * load_penalty)
}
```

Fallback when all workers are excluded:

```rust
workers.iter()
    .filter(|w| w.is_healthy())
    .min_by(|a, b| {
        a.kv_cache_utilization()
            .partial_cmp(&b.kv_cache_utilization())
            .unwrap()
    })
```

Extend `on_request_complete` to carry the token count so `decrement_load_with_tokens` is accurate:

```rust
fn on_request_complete(&self, worker_url: &str, success: bool, tokens: usize);
```

---

## PR Breakdown

| PR | Phases | Description |
|---|---|---|
| PR 1 | 1 + 2 | Worker state extensions + KV async refresh. No behavior change. |
| PR 2 | 3 + 4 | `/tokenize` integration + token-based radix tree rewrite. |
| PR 3 | 5 + 6 | Config extensions + new scoring formula. Actual behavior change. |

Each PR is independently reviewable and deployable. During rollout, `lambda = 0` in config reproduces pre-PR-3 cache-only behavior.

---

## Files Affected

| File | Change |
|---|---|
| `src/core/worker.rs` | New fields + trait methods for tokens and KV utilization |
| `src/tree.rs` | Node text `String` → `Vec<u32>`, all traversal/insert/evict logic |
| `src/policies/cache_aware.rs` | `/tokenize` call, new scoring formula, fallback logic |
| `src/policies/mod.rs` | `CacheAwareConfig` new fields, `on_request_complete` signature |
| `src/config/types.rs` | New config fields wired through |

---

## Open Questions (Resolved)

| Question | Decision |
|---|---|
| Which worker to tokenize on? | Randomly selected healthy worker for the model |
| Tokenization cache? | None — intra-cluster latency is low enough |
| Token tree in same PR as tokenize? | Yes (Phase 3 + 4 together) |
| `prompt_token_ids` support in vLLM? | Confirmed — running v0.18.0 |
| KV stats: background thread vs inline? | Short-TTL async refresh, demand-driven, no dedicated thread |
