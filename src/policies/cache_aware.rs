/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. First, check for empty pods (load == 0) and prioritize them:
       Select the empty pod with the smallest tree size (most available cache capacity in terms of cached data)
       This gives empty pods with less cached data a chance to build up their cache
    b. If no empty pods, find the worker with the highest prefix match for the request
    c. If match rate > cache_threshold:
       Route to the worker with highest match (likely has relevant data cached)
    d. If match rate ≤ cache_threshold:
       Route to the worker with smallest tree size (most available cache capacity)
    e. Background maintenance:
       Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::Tree;
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{debug, info};

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>, // model_id -> Arc<Tree>
    eviction_handle: Option<thread::JoinHandle<()>>,
    /// Set to true on Drop to signal the eviction thread to exit cleanly.
    shutdown: Arc<AtomicBool>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        // Start background eviction thread if configured.
        // The thread sleeps in 500 ms ticks, checking the shutdown flag each tick so
        // it exits promptly when the policy is dropped — avoiding a memory leak from
        // the eviction thread pinning the `trees` Arc indefinitely.
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let shutdown_clone = Arc::clone(&shutdown);
            let max_tree_size = config.max_tree_size;
            let interval_secs = config.eviction_interval_secs;

            Some(thread::spawn(move || {
                let tick = Duration::from_millis(500);
                let full_interval = Duration::from_secs(interval_secs);
                let mut elapsed = Duration::ZERO;

                loop {
                    thread::sleep(tick);

                    if shutdown_clone.load(Ordering::Acquire) {
                        break;
                    }

                    elapsed += tick;
                    if elapsed >= full_interval {
                        elapsed = Duration::ZERO;

                        // Evict for all model trees
                        for entry in trees_clone.iter() {
                            let model_id = entry.key();
                            let tree = entry.value();
                            tree.evict_tenant_by_size(max_tree_size);
                            debug!(
                                "Cache eviction completed for model {}, max_size: {}",
                                model_id, max_tree_size
                            );
                        }
                    }
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            eviction_handle,
            shutdown,
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
    }

    /// Remove a worker from the tree.
    ///
    /// The `DashMap::get` ref is cloned out and dropped BEFORE `remove` is
    /// called to avoid a self-deadlock: holding a read-guard on a shard while
    /// calling `remove` on the same shard would attempt to acquire a write-lock
    /// on an already-read-locked shard.
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        // Clone the Arc<Tree> out; the DashMap Ref (shard read-guard) is
        // dropped at the end of this statement, before any further operations.
        let tree = self.trees.get(tree_key).map(|e| e.value().clone());
        if let Some(tree) = tree {
            tree.remove_tenant(worker.url());
            // Use remove_if for atomic check-and-remove: the predicate runs
            // under the shard write-lock, so a concurrent add_worker's
            // entry().or_insert_with() on the same shard serializes correctly.
            // If a worker was just re-added the predicate sees a non-empty tree
            // and skips removal; if removed first, add_worker recreates the tree.
            self.trees.remove_if(tree_key, |_, t| t.is_empty());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility).
    ///
    /// Collects keys of trees that became empty during the iteration, then
    /// removes them AFTER the iterator is done to avoid mutating the map
    /// while iterating over it.
    pub fn remove_worker_by_url(&self, url: &str) {
        let mut empty_keys: Vec<String> = Vec::new();

        // Iterate to remove the tenant and record which trees are now empty.
        // Do NOT call self.trees.remove() inside this loop — that would mutate
        // the DashMap while an iterator holds a shard guard, which can deadlock.
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
            if tree_ref.value().is_empty() {
                empty_keys.push(tree_ref.key().clone());
            }
        }

        // Iterator guard is fully released here; use remove_if for atomic
        // check-and-remove so a concurrent add_worker that re-inserted a tenant
        // between the is_empty() check above and the remove below is not lost.
        for k in empty_keys {
            self.trees.remove_if(&k, |_, t| t.is_empty());
        }
    }

    /// Explicitly remove the tree for a model that has been fully decommissioned.
    ///
    /// The model key is normalised with `normalize_model_key` so callers can
    /// pass the raw model ID as reported by workers.
    pub fn remove_model(&self, model_id: &str) {
        let key = normalize_model_key(model_id);
        self.trees.remove(key);
    }

    /// Return true if the tree map contains an entry for `model_id`.
    ///
    /// Available only in test builds so external test modules can assert on
    /// tree lifecycle without accessing the private `trees` field.
    /// The model_id is normalized via `normalize_model_key` so callers can
    /// pass un-normalized IDs without getting misleading false negatives.
    #[cfg(test)]
    pub fn has_tree_for_model(&self, model_id: &str) -> bool {
        let key = normalize_model_key(model_id);
        self.trees.contains_key(key)
    }

    /// Return true if `worker_url` is present in the tree for `model_id`.
    ///
    /// Available only in test builds.
    /// The model_id is normalized via `normalize_model_key` so callers can
    /// pass un-normalized IDs without getting misleading false negatives.
    #[cfg(test)]
    pub fn tree_has_tenant(&self, model_id: &str, worker_url: &str) -> bool {
        let key = normalize_model_key(model_id);
        self.trees
            .get(key)
            .map(|e| e.value().tenant_char_count.contains_key(worker_url))
            .unwrap_or(false)
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
        RouterMetrics::record_load_balancing_event();
        RouterMetrics::set_load_range(max_load, min_load);

        // Feature 2: Enhanced Load Calculation
        // Filter workers: exclude those with KV util > threshold (unless all are > threshold)
        let kv_threshold = self.config.kv_util_threshold;
        let empty_pods: Vec<usize> = healthy_indices
            .iter()
            .filter(|&&idx| {
                workers[idx].load() == 0 && workers[idx].kv_cache_utilization() <= kv_threshold
            })
            .copied()
            .collect();

        // Get the normalized model key for tree lookup
        let normalized_model_id = normalize_model_key(model_id);

        // Snapshot metrics for each empty pod once to avoid repeated DashMap lookups
        // and to get a consistent view when values change mid-sort.
        // `rand_key` is drawn once per candidate so the comparator is a valid total
        // order (transitive) while still producing a uniform random choice among
        // exactly-tied candidates.
        struct WorkerSnapshot {
            idx: usize,
            kv_util: f32,
            load: usize,
            tree_size: usize,
            rand_key: u32,
        }

        // Clone the Arc<Tree> out of the DashMap so the shard read-guard (Ref)
        // is released before the snapshot loop runs.  Holding a DashMap Ref
        // across arbitrary work (including nested DashMap reads inside
        // tenant_char_count) risks a deadlock if a concurrent tree removal
        // tries to acquire a write-lock on the same shard.
        let tree_arc: Option<Arc<Tree>> = self.trees.get(normalized_model_id).map(|e| e.value().clone());

        let selected_idx = if !empty_pods.is_empty() {
            // Prioritize empty pods: pick one with smallest tree size among them.
            // Snapshot tree sizes and a stable random tiebreak key once per worker.
            let snapshots: Vec<WorkerSnapshot> = empty_pods
                .iter()
                .map(|&idx| {
                    let tree_size = tree_arc
                        .as_ref()
                        .and_then(|tree| {
                            tree.tenant_char_count
                                .get(workers[idx].url())
                                .map(|entry| *entry.value())
                        })
                        .unwrap_or(0);
                    let rand_key: u32 = rand::rng().random::<u32>();
                    WorkerSnapshot {
                        idx,
                        kv_util: 0.0,
                        load: 0,
                        tree_size,
                        rand_key,
                    }
                })
                .collect();

            snapshots
                .iter()
                .min_by_key(|s| (s.tree_size, s.rand_key))
                .map(|s| s.idx)
        } else {
            // If all workers exceed KV threshold, use all healthy workers
            let filtered_indices: Vec<usize> = healthy_indices
                .iter()
                .filter(|&&idx| workers[idx].kv_cache_utilization() <= kv_threshold)
                .copied()
                .collect();

            if filtered_indices.is_empty() {
                debug!(
                    "All workers exceed KV utilization threshold {}, using all healthy workers",
                    kv_threshold
                );

                // Snapshot tree sizes and stable random tiebreak keys for all
                // healthy workers, using the Arc<Tree> cloned above.
                let snapshots: Vec<WorkerSnapshot> = healthy_indices
                    .iter()
                    .map(|&idx| {
                        let tree_size = tree_arc
                            .as_ref()
                            .and_then(|tree| {
                                tree.tenant_char_count
                                    .get(workers[idx].url())
                                    .map(|entry| *entry.value())
                            })
                            .unwrap_or(0);
                        let rand_key: u32 = rand::rng().random::<u32>();
                        WorkerSnapshot {
                            idx,
                            kv_util: 0.0,
                            load: 0,
                            tree_size,
                            rand_key,
                        }
                    })
                    .collect();

                snapshots
                    .iter()
                    .min_by_key(|s| (s.tree_size, s.rand_key))
                    .map(|s| s.idx)
            } else {
                // Compute score = alpha * kv_util + beta * normalized_load for each candidate.
                // Snapshot all metrics once per worker to avoid repeated trait calls and
                // to get a consistent ordering when values shift mid-sort (O(n log n) comparisons
                // would otherwise re-read live values).
                let alpha = self.config.alpha;
                let beta = self.config.beta;

                let snapshots: Vec<WorkerSnapshot> = filtered_indices
                    .iter()
                    .map(|&idx| {
                        let kv_util = workers[idx].kv_cache_utilization();
                        let load = workers[idx].load();
                        let rand_key: u32 = rand::rng().random::<u32>();
                        WorkerSnapshot {
                            idx,
                            kv_util,
                            load,
                            tree_size: 0,
                            rand_key,
                        }
                    })
                    .collect();

                snapshots
                    .iter()
                    .min_by(|a, b| {
                        let norm_load_a = (a.load as f32 / 100.0).min(1.0);
                        let norm_load_b = (b.load as f32 / 100.0).min(1.0);

                        let score_a = alpha * a.kv_util + beta * norm_load_a;
                        let score_b = alpha * b.kv_util + beta * norm_load_b;

                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.rand_key.cmp(&b.rand_key))
                    })
                    .map(|s| s.idx)
            }
        };

        let min_load_idx = selected_idx?;

        debug!(
            "Selected worker {} with load {}",
            workers[min_load_idx].url(),
            workers[min_load_idx].load()
        );

        // Even in imbalanced mode, update the tree to maintain cache state
        if let Some(text) = request_text {
            // Get the tree reference without locking the entire HashMap
            // DashMap only locks the specific shard containing this key
            let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                tree.insert(text, worker_url);
            } else {
                debug!(
                    "Warning: No tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[min_load_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

        Some(min_load_idx)
    }

    /// Helper method to estimate token count from request text
    /// Uses simple heuristic: characters / 4
    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Feature 1: Small Request Bypass
        // If request is small (tokens < 25000), use enhanced load-based selection (select_worker_min_load)
        // instead of simple least-loaded selection
        if let Some(text) = request_text {
            let estimated_tokens = self.estimate_tokens(text);
            if estimated_tokens < self.config.small_request_token_threshold {
                debug!(
                    "Small request bypass (< {} tokens): estimated {} tokens, using select_worker_min_load",
                    self.config.small_request_token_threshold, estimated_tokens
                );
                // Compute load stats for select_worker_min_load
                let (min_load, max_load) =
                    workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
                        let load = w.load();
                        (min.min(load), max.max(load))
                    });
                let min_load = if min_load == usize::MAX { 0 } else { min_load };
                let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

                return self.select_worker_min_load(
                    workers,
                    request_text,
                    &healthy_indices,
                    &model_id,
                    max_load,
                    min_load,
                );
            }
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Get current load statistics - compute min/max in single pass without allocation
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        debug!(
            "Load status for model: max_load={}, min_load={}, is_imbalanced={}",
            max_load, min_load, is_imbalanced
        );

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                request_text,
                &healthy_indices,
                model_id,
                max_load,
                min_load,
            );
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        // Only collect tree keys when debug logging is actually enabled to avoid
        // a full DashMap iteration + Vec allocation on every request.
        if tracing::enabled!(tracing::Level::DEBUG) {
            let keys: Vec<_> = self.trees.iter().map(|entry| entry.key().clone()).collect();
            debug!("Available tree keys: {:?}", keys);
        }

        let Some(tree) = tree else {
            // No tree for this model, log warning and use random selection
            debug!(
                "Warning: No tree found for model '{}', using random worker selection",
                model_id
            );
            // Return a random healthy worker
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            let selected_idx = healthy_indices[random_idx];

            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[selected_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

            return Some(selected_idx);
        };
        debug!("Using cache-aware routing for model '{}'", model_id);
        // Now we work with the tree without holding the HashMap lock
        // Use prefix_match_with_counts to avoid redundant chars().count() calls
        let result = tree.prefix_match_with_counts(text);
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        debug!(
            "Cache match for model '{}': matched_chars={}, input_chars={}, match_rate={:.2}",
            model_id, result.matched_char_count, result.input_char_count, match_rate
        );

        // Feature 3: Cache Threshold Fallback
        // Select worker based on cache match rate
        let selected_idx = if match_rate > self.config.cache_threshold {
            // Cache hit path: find worker by URL
            let tenant_url: &str = &result.tenant;
            workers
                .iter()
                .position(|w| w.url() == tenant_url)
                .filter(|&idx| {
                    // Check if target pod meets the criteria
                    let kv_util = workers[idx].kv_cache_utilization();
                    let load = workers[idx].load();
                    let kv_ok = kv_util < self.config.kv_util_threshold;
                    let load_ok = load < self.config.balance_abs_threshold;

                    if !kv_ok || !load_ok {
                        debug!(
                            "Cache target worker {} fails fallback check: kv_util={:.2} (threshold={:.2}), load={} (threshold={})",
                            workers[idx].url(), kv_util, self.config.kv_util_threshold, load, self.config.balance_abs_threshold
                        );
                    }

                    workers[idx].is_healthy() && kv_ok && load_ok
                })
                .or_else(|| {
                    // Fallback to min_load selection if target pod doesn't meet criteria
                    debug!("Cache target fails criteria, falling back to min_load selection");
                    self.select_worker_min_load(
                        workers,
                        request_text,
                        &healthy_indices,
                        model_id,
                        max_load,
                        min_load,
                    )
                })
        } else {
            // Low cache match: use worker with minimum load
            self.select_worker_min_load(
                workers,
                request_text,
                &healthy_indices,
                model_id,
                max_load,
                min_load,
            )
        };

        if let Some(idx) = selected_idx {
            // Update the tree with this request (use worker URL directly, no allocation)
            tree.insert(text, workers[idx].url());
            debug!("Inserted tree in worker {}", workers[idx].url());

            // Increment processed counter
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            return Some(idx);
        }

        // Selected worker no longer exists or unhealthy, remove stale tenant from tree
        if match_rate > self.config.cache_threshold {
            let tenant_url: &str = &result.tenant;
            tree.remove_tenant(tenant_url);
            debug!("Removed stale worker {} from cache tree", tenant_url);
        }

        // Fallback to first healthy worker
        if let Some(idx) = healthy_indices.first().copied() {
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            Some(idx)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // DEPRECATED: This method is no longer used when separate policies are configured.
        // The PD router now uses separate policies for prefill and decode selection.
        // This implementation remains for backward compatibility when a single policy is used.

        // In PD mode with single policy:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }

    fn requires_initialization(&self) -> bool {
        true // Cache-aware policy requires init_workers() to set up trees
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        info!(
            "Initializing workers for cache-aware policy: {}",
            workers
                .iter()
                .map(|w| w.url())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            info!(
                "Creating tree for model key: '{}' with {} workers",
                tree_key,
                model_workers.len()
            );
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Signal the eviction thread to exit, then join it so it releases its
        // `trees` Arc and terminates cleanly — avoiding both the memory leak and
        // the zombie thread from the previous infinite-loop design.
        self.shutdown.store(true, Ordering::Release);
        if let Some(h) = self.eviction_handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread and without small-request bypass so
        // that "hello world" (< 25000 tokens) exercises the cache-aware path, not
        // the min-load path. The test checks cache affinity: two identical requests
        // must land on the same worker.
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,          // Disable eviction thread
            small_request_token_threshold: 0,   // Disable small-request bypass
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker_with_headers(&workers, Some("hello world"), None)
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker_with_headers(&workers, Some("hello world"), None)
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker_with_headers(&workers, Some("hello"), None)
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
            small_request_token_threshold: 25000,
            kv_util_threshold: 0.9,
            alpha: 0.7,
            beta: 0.3,
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy
                .select_worker_with_headers(&workers, Some("test"), None)
                .unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker_with_headers(&workers, Some("test1"), None);
        policy.select_worker_with_headers(&workers, Some("test2"), None);

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy
            .select_worker_with_headers(&workers, Some("test1"), None)
            .unwrap();
        assert_eq!(idx, 1);
    }

    /// When all workers for a model are removed the model's tree must be
    /// cleaned out of `self.trees` to prevent unbounded memory growth.
    #[test]
    fn test_tree_removed_when_last_worker_removed() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            small_request_token_threshold: 0, // keep test deterministic
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        // Use a custom model ID so we can assert on it directly.
        // The model_id is stored as a label ("model_id" key) on the worker.
        let mut labels = std::collections::HashMap::new();
        labels.insert("model_id".to_string(), "model-alpha".to_string());

        let w1 = BasicWorker::new("http://m1w1:8000".to_string(), WorkerType::Regular)
            .with_labels(labels.clone());
        let w2 = BasicWorker::new("http://m1w2:8000".to_string(), WorkerType::Regular)
            .with_labels(labels.clone());

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Confirm the tree was created.
        assert!(
            policy.trees.contains_key("model-alpha"),
            "tree must exist after init"
        );

        // Remove both workers.
        policy.remove_worker(workers[0].as_ref());
        policy.remove_worker(workers[1].as_ref());

        // The tree entry must have been cleaned up.
        assert!(
            !policy.trees.contains_key("model-alpha"),
            "tree must be removed after last worker is gone"
        );
    }

    /// When all workers for *one* model are removed the *other* model's tree
    /// must remain intact.
    #[test]
    fn test_other_model_tree_survives_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            small_request_token_threshold: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let mut alpha_labels = std::collections::HashMap::new();
        alpha_labels.insert("model_id".to_string(), "model-alpha".to_string());

        let mut beta_labels = std::collections::HashMap::new();
        beta_labels.insert("model_id".to_string(), "model-beta".to_string());

        let alpha_w = BasicWorker::new("http://alpha:8000".to_string(), WorkerType::Regular)
            .with_labels(alpha_labels);
        let beta_w = BasicWorker::new("http://beta:8000".to_string(), WorkerType::Regular)
            .with_labels(beta_labels);

        let all_workers: Vec<Arc<dyn Worker>> = vec![Arc::new(alpha_w), Arc::new(beta_w)];
        policy.init_workers(&all_workers);

        assert!(policy.trees.contains_key("model-alpha"));
        assert!(policy.trees.contains_key("model-beta"));

        // Remove only model-alpha's worker.
        policy.remove_worker(all_workers[0].as_ref());

        assert!(
            !policy.trees.contains_key("model-alpha"),
            "model-alpha tree must be gone"
        );
        assert!(
            policy.trees.contains_key("model-beta"),
            "model-beta tree must remain"
        );
    }

    // ===== Repro: "This is a cat" / "This is a dog" cache behavior =====

    /// The radix tree itself correctly caches BOTH divergent-suffix strings.
    /// This exonerates the tree: it is NOT dropping "This is a dog".
    #[test]
    fn repro_tree_caches_both_divergent_prefixes() {
        let tree = Tree::new();
        tree.insert("This is a cat", "w1");
        tree.insert("This is a dog", "w2");

        let r1 = tree.prefix_match_with_counts("This is a cat");
        assert_eq!(r1.matched_char_count, 13, "cat should fully match");
        assert_eq!(&*r1.tenant, "w1");

        let r2 = tree.prefix_match_with_counts("This is a dog");
        assert_eq!(r2.matched_char_count, 13, "dog should fully match — it IS cached");
        assert_eq!(&*r2.tenant, "w2");

        // A third divergent suffix shares the 10-char "This is a " prefix.
        let r3 = tree.prefix_match_with_counts("This is a fox");
        assert_eq!(r3.matched_char_count, 10, "shared 'This is a ' prefix");
    }

    fn two_workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ]
    }

    fn cfg(small_request_token_threshold: usize) -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 0,
            max_tree_size: 100_000,
            small_request_token_threshold,
            kv_util_threshold: 0.9,
            alpha: 0.7,
            beta: 0.3,
        }
    }

    /// With the DEFAULT threshold (25000), both tiny prompts hit the small-request
    /// bypass → load-based selection → they are spread to DIFFERENT workers, so the
    /// shared "This is a " prefix cached by req1 is NOT reused by req2.
    #[test]
    fn repro_small_request_bypass_breaks_affinity() {
        let policy = CacheAwarePolicy::with_config(cfg(25_000));
        let workers = two_workers();
        policy.init_workers(&workers);

        let idx1 = policy
            .select_worker_with_headers(&workers, Some("This is a cat"), None)
            .unwrap();
        let idx2 = policy
            .select_worker_with_headers(&workers, Some("This is a dog"), None)
            .unwrap();

        assert_ne!(
            idx1, idx2,
            "default 25000-token bypass spreads tiny prompts across workers (no cache affinity)"
        );
    }

    /// With the bypass DISABLED (threshold 0), the cache-aware prefix path runs and
    /// req2 follows req1 to the SAME worker via the shared "This is a " prefix.
    #[test]
    fn repro_affinity_works_when_bypass_disabled() {
        let policy = CacheAwarePolicy::with_config(cfg(0));
        let workers = two_workers();
        policy.init_workers(&workers);

        let idx1 = policy
            .select_worker_with_headers(&workers, Some("This is a cat"), None)
            .unwrap();
        let idx2 = policy
            .select_worker_with_headers(&workers, Some("This is a dog"), None)
            .unwrap();

        assert_eq!(
            idx1, idx2,
            "with bypass off, shared prefix routes req2 to req1's worker (cache hit)"
        );
    }
}
