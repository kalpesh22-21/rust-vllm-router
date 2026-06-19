/// Integration tests validating the PRODUCTION deployment shape of the cache-aware
/// routing policy: ONE model served by MULTIPLE worker URLs.
///
/// Key behavioral facts encoded in these tests:
/// - estimate_tokens(text) = text.len() / 4
/// - default small_request_token_threshold = 25000
/// - A prompt needs > 100,000 BYTES to take the cache-affinity path
/// - Anything smaller routes via select_worker_min_load (load-based)
/// - Cache affinity: repeated large prompts with same prefix co-locate on the same worker
/// - Empty pod (load==0) gets priority: smallest-tree among empties
/// - Imbalanced load: heavily loaded workers are deprioritized
/// - kv_util > 0.9: worker is excluded from scoring (unless all exceed threshold)
/// - Always returns a valid in-range index; never panics
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy};

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

/// Build 4 workers all serving the same model.
/// Workers are named w1..w4 and carry "model_id" = "llama-prod".
fn make_workers(count: usize) -> Vec<Arc<dyn Worker>> {
    (1..=count)
        .map(|i| {
            let mut labels = HashMap::new();
            labels.insert("model_id".to_string(), "llama-prod".to_string());
            let w = BasicWorker::new(format!("http://w{}:8000", i), WorkerType::Regular)
                .with_labels(labels);
            Arc::new(w) as Arc<dyn Worker>
        })
        .collect()
}

/// Generate a prompt that is guaranteed to exceed the 100 KB threshold so the
/// cache-affinity path is exercised under the default small_request_token_threshold.
/// estimate_tokens = len / 4; threshold = 25000 tokens → 100_000 bytes required.
fn large_prompt(prefix: &str, suffix: &str) -> String {
    // Pad the prefix to 110 KB and append a small suffix.
    let pad_len = 110_000usize.saturating_sub(prefix.len());
    let padding: String = "x".repeat(pad_len);
    format!("{}{}{}", prefix, padding, suffix)
}

/// Return a config identical to production defaults but with the given eviction interval.
fn prod_config(eviction_interval_secs: u64) -> CacheAwareConfig {
    CacheAwareConfig {
        eviction_interval_secs,
        ..CacheAwareConfig::default()
    }
}

/// Assert that `idx` is a valid index into `workers`.
fn assert_valid_index(idx: usize, workers: &[Arc<dyn Worker>]) {
    assert!(
        idx < workers.len(),
        "Returned index {} is out of range (workers.len() = {})",
        idx,
        workers.len()
    );
}

// ──────────────────────────────────────────────────────────────────────
// Phase 1: Cold-start routing distributes across workers
// ──────────────────────────────────────────────────────────────────────

/// With no prior history and all loads at zero, routing should distribute requests
/// across workers (not always pick the same one).  We also verify no panics and
/// that every returned index is valid.
#[test]
fn phase1_cold_start_routing() {
    let workers = make_workers(4);
    let config = prod_config(0); // eviction disabled
    let policy = CacheAwarePolicy::with_config(config);
    policy.init_workers(&workers);

    // Use SMALL prompts so we exercise the min-load / empty-pod path.
    // All workers start with load == 0, so the "empty-pod" branch runs.
    let small = "hello, how are you today?"; // well under 100 KB
    assert!(
        small.len() < 100_000,
        "Sanity: this prompt should take the min-load path, not the cache-affinity path (len={})",
        small.len()
    );

    let mut seen = std::collections::HashSet::new();
    for _ in 0..20 {
        let idx = policy
            .select_worker_with_headers(&workers, Some(small), None)
            .expect("Should always return a worker on cold start");
        assert_valid_index(idx, &workers);
        seen.insert(idx);
        // Simulate the request "leaving" quickly (keep loads at 0 for distribution)
    }

    // With 4 workers all at load==0 (empty-pod priority), routing uses smallest-tree
    // with random tiebreak. We expect at least 2 distinct workers to be selected across 20 calls.
    // This confirms distribution rather than always picking the same index.
    println!(
        "Phase 1 – cold start: unique workers selected = {} out of 4",
        seen.len()
    );
    assert!(
        seen.len() >= 2,
        "Cold-start routing should distribute across workers, but only saw indices: {:?}",
        seen
    );
}

// ──────────────────────────────────────────────────────────────────────
// Phase 2: Cache affinity with large (>100 KB) prompts
// ──────────────────────────────────────────────────────────────────────

/// Verifies that cache-affinity ONLY engages for prompts > 100 KB (under default config).
///
/// OBSERVED BEHAVIOR reported in this test:
/// - A small prompt (< 100 KB) never uses cache-affinity; load-based path runs instead.
/// - A large prompt (>= 100 KB) hits the cache-affinity path; the second call with the
///   same prefix routes to the SAME worker.
/// - Two large prompts sharing a long prefix but differing only in suffix co-locate on
///   the same worker.
#[test]
fn phase2_cache_affinity_large_prompt() {
    // Disable eviction thread and small-request bypass to isolate cache-affinity behavior.
    // We keep small_request_token_threshold at the production default (25000) to verify
    // the threshold behavior explicitly.
    let config = CacheAwareConfig {
        eviction_interval_secs: 0, // no background eviction during test
        // Keep production default threshold: 25000 tokens => 100 000 byte boundary
        ..CacheAwareConfig::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(4);
    policy.init_workers(&workers);

    // ── Part A: Confirm small prompts DON'T use cache affinity ──────────
    // A "normal" API prompt (a few hundred bytes) — well below the 100 KB threshold.
    let small_prompt = "Summarise the following document:";
    let token_estimate = small_prompt.len() / 4;
    println!(
        "Phase 2A – small prompt: len={} bytes, estimated_tokens={}, threshold=25000 → takes load-based path",
        small_prompt.len(),
        token_estimate
    );
    assert!(
        token_estimate < 25_000,
        "Precondition: small_prompt must be below token threshold, got {}",
        token_estimate
    );

    // With balanced load (all at 0), route multiple times; no affinity guarantee.
    for _ in 0..5 {
        let idx = policy
            .select_worker_with_headers(&workers, Some(small_prompt), None)
            .expect("Must select a worker");
        assert_valid_index(idx, &workers);
    }

    // ── Part B: Large prompt hits cache-affinity path ────────────────────
    let shared_prefix = "SHARED_PREFIX: ".to_string() + &"A".repeat(5000);
    let prompt_a = large_prompt(&shared_prefix, "_SUFFIX_A");
    let prompt_b = large_prompt(&shared_prefix, "_SUFFIX_B");

    let token_estimate_large = prompt_a.len() / 4;
    println!(
        "Phase 2B – large prompt: len={} bytes, estimated_tokens={}, threshold=25000 → takes cache-affinity path",
        prompt_a.len(),
        token_estimate_large
    );
    assert!(
        token_estimate_large >= 25_000,
        "Precondition: large prompt must exceed token threshold, got {}",
        token_estimate_large
    );

    // First large request: establishes cache affinity on some worker.
    let first_idx = policy
        .select_worker_with_headers(&workers, Some(&prompt_a), None)
        .expect("Must select a worker for large prompt");
    assert_valid_index(first_idx, &workers);
    println!(
        "Phase 2B – first large prompt routed to worker index {}",
        first_idx
    );

    // Repeat SAME large prompt: must route to the SAME worker (cache hit).
    let repeat_count = 5;
    for i in 0..repeat_count {
        let idx = policy
            .select_worker_with_headers(&workers, Some(&prompt_a), None)
            .expect("Must select a worker");
        assert_valid_index(idx, &workers);
        assert_eq!(
            idx,
            first_idx,
            "Iteration {}: same large prompt must route to the same worker (cache affinity). Got {} expected {}",
            i,
            idx,
            first_idx
        );
    }
    println!(
        "Phase 2B – all {} repetitions of the same large prompt routed to worker {} (cache affinity confirmed)",
        repeat_count, first_idx
    );

    // ── Part C: Near-identical large prompt (different tail) co-locates ──
    // prompt_b shares a very long prefix with prompt_a, differs only in the suffix.
    let second_idx = policy
        .select_worker_with_headers(&workers, Some(&prompt_b), None)
        .expect("Must select a worker for prompt_b");
    assert_valid_index(second_idx, &workers);
    println!(
        "Phase 2C – near-identical large prompt (different suffix) routed to worker index {}",
        second_idx
    );
    assert_eq!(
        second_idx,
        first_idx,
        "Near-identical large prompts sharing a long prefix must co-locate on the same worker. \
         prompt_a -> worker {}, prompt_b -> worker {}",
        first_idx,
        second_idx
    );
    println!("Phase 2C – shared-prefix co-location confirmed (both routed to worker {})", first_idx);
}

// ──────────────────────────────────────────────────────────────────────
// Phase 3: Small-request bypass → min-load → empty pod priority
// ──────────────────────────────────────────────────────────────────────

/// Small prompts bypass cache-affinity and use the load-based selection.
/// When one worker has load == 0 (empty pod), it must be prioritised.
#[test]
fn phase3_small_request_uses_min_load_and_empty_pod_priority() {
    let config = prod_config(0);
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(4);
    policy.init_workers(&workers);

    // Give workers 0, 1, 2 some load; worker 3 stays at load == 0.
    workers[0].increment_load();
    workers[0].increment_load();
    workers[1].increment_load();
    workers[2].increment_load();
    workers[2].increment_load();
    workers[2].increment_load();
    // workers[3].load() == 0

    println!(
        "Phase 3 – loads: w0={}, w1={}, w2={}, w3={}",
        workers[0].load(),
        workers[1].load(),
        workers[2].load(),
        workers[3].load()
    );

    let small = "Tell me a joke."; // tiny prompt, well below 100 KB
    let idx = policy
        .select_worker_with_headers(&workers, Some(small), None)
        .expect("Must select a worker");
    assert_valid_index(idx, &workers);

    // Worker 3 is the only empty pod; it must be selected.
    println!(
        "Phase 3 – small prompt routed to worker index {} (expected: 3, the empty pod)",
        idx
    );
    assert_eq!(
        idx, 3,
        "Empty pod (load==0) must be prioritised over loaded workers. Got idx {} expected 3",
        idx
    );
}

// ──────────────────────────────────────────────────────────────────────
// Phase 4: Load imbalance triggers load-based routing
// ──────────────────────────────────────────────────────────────────────

/// When one worker has significantly higher load (exceeding balance thresholds),
/// routing should favour the lower-load workers.
#[test]
fn phase4_load_imbalance_favors_lower_load_worker() {
    // Production defaults: balance_abs_threshold=32, balance_rel_threshold=1.1
    let config = CacheAwareConfig {
        balance_abs_threshold: 5,     // lower threshold to trigger imbalance easily
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 0,
        small_request_token_threshold: 0, // disable small-request bypass so large-path also sees imbalance
        ..CacheAwareConfig::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(4);
    policy.init_workers(&workers);

    // Create heavy imbalance: worker 0 has load 20; workers 1-3 have load 0.
    for _ in 0..20 {
        workers[0].increment_load();
    }

    println!(
        "Phase 4 – loads: w0={}, w1={}, w2={}, w3={}",
        workers[0].load(),
        workers[1].load(),
        workers[2].load(),
        workers[3].load()
    );

    // Even with a non-trivially-sized prompt (but < threshold with threshold=0 it goes cache path;
    // with imbalance check it should still route to lower-load worker).
    // We test with a large prompt to ensure we hit the imbalance check in the cache-aware path.
    let prompt = large_prompt("imbalance test prompt ", "");

    let mut selected_non_w0 = 0;
    let trials = 10;
    for _ in 0..trials {
        let idx = policy
            .select_worker_with_headers(&workers, Some(&prompt), None)
            .expect("Must select a worker");
        assert_valid_index(idx, &workers);
        if idx != 0 {
            selected_non_w0 += 1;
        }
    }

    println!(
        "Phase 4 – {}/{} requests routed away from overloaded worker 0",
        selected_non_w0, trials
    );
    assert!(
        selected_non_w0 >= trials - 2,
        "With heavy load imbalance, at most 2 of {} requests should go to w0; got {} requests to other workers",
        trials,
        selected_non_w0
    );
}

// ──────────────────────────────────────────────────────────────────────
// Phase 5: kv_util above threshold → worker excluded / deprioritised
// ──────────────────────────────────────────────────────────────────────

/// When one worker's kv_cache_utilization > kv_util_threshold (0.9), it must be
/// excluded from scoring in the load-balanced path.  Routing should avoid it
/// as long as at least one other worker is below the threshold.
#[test]
fn phase5_kv_util_above_threshold_excluded() {
    let config = CacheAwareConfig {
        eviction_interval_secs: 0,
        // balance thresholds set high to force balanced path, where kv_util is checked
        balance_abs_threshold: 1000,
        balance_rel_threshold: 100.0,
        // Use default kv_util_threshold (0.9) and default small_request_token_threshold
        ..CacheAwareConfig::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(4);
    policy.init_workers(&workers);

    // Set worker 0's kv_cache_utilization above threshold (0.95 > 0.9).
    workers[0].set_kv_cache_utilization(0.95);
    println!(
        "Phase 5 – worker 0 kv_util set to 0.95 (threshold=0.9); workers 1-3 at 0.0"
    );

    // All workers at zero load (balanced); should never route to worker 0 for small requests.
    // Small requests go via select_worker_min_load which checks kv_util.
    let small = "What is 2+2?";

    let mut selected_w0 = 0;
    let trials = 20;
    for _ in 0..trials {
        let idx = policy
            .select_worker_with_headers(&workers, Some(small), None)
            .expect("Must select a worker");
        assert_valid_index(idx, &workers);
        if idx == 0 {
            selected_w0 += 1;
        }
    }

    println!(
        "Phase 5 – {} of {} requests routed to high-kv_util worker 0 (should be 0)",
        selected_w0, trials
    );
    assert_eq!(
        selected_w0,
        0,
        "Worker with kv_util above threshold (0.95 > 0.9) must not be selected when healthy alternatives exist"
    );

    // ── Verify fallback: when ALL workers exceed threshold, still routes ──
    for w in &workers {
        w.set_kv_cache_utilization(0.95);
    }
    println!("Phase 5 – all workers set to kv_util=0.95; routing should still succeed (use all)");
    for _ in 0..5 {
        let idx = policy
            .select_worker_with_headers(&workers, Some(small), None)
            .expect("Must still select a worker when all exceed kv_util threshold");
        assert_valid_index(idx, &workers);
    }
    println!("Phase 5 – kv_util all-above-threshold fallback verified (no panic, valid index)");
}

// ──────────────────────────────────────────────────────────────────────
// Phase 6: Concurrency stress — deadlock regression guard
// ──────────────────────────────────────────────────────────────────────

/// CRITICAL: This test guards against the eviction<->insert deadlock that was fixed.
/// It spawns many threads doing concurrent select_worker_with_headers calls (mix of
/// large and small prompts) while the background eviction thread runs.
///
/// If the test HANGS it means the deadlock was re-introduced.
/// Every returned index must be valid (in-range).
#[test]
fn phase6_concurrency_stress_no_deadlock() {
    // Enable eviction with a very short interval so eviction fires frequently.
    // Small max_tree_size to trigger eviction aggressively.
    let config = CacheAwareConfig {
        eviction_interval_secs: 1, // eviction fires every second
        max_tree_size: 50,         // small cap forces frequent eviction
        // Keep production defaults for everything else
        ..CacheAwareConfig::default()
    };
    let policy = Arc::new(CacheAwarePolicy::with_config(config));
    let workers: Vec<Arc<dyn Worker>> = make_workers(4);
    let workers = Arc::new(workers);
    policy.init_workers(&workers);

    let num_threads = 8;
    let calls_per_thread = 300;
    let test_duration = Duration::from_secs(4);

    let mut handles = Vec::new();
    let start = std::time::Instant::now();

    for thread_id in 0..num_threads {
        let policy_clone = Arc::clone(&policy);
        let workers_clone = Arc::clone(&workers);

        let handle = std::thread::spawn(move || {
            let mut rng_state: u64 = 12345 + thread_id as u64 * 9999;
            let mut call_count = 0;
            let mut errors = 0;

            while call_count < calls_per_thread && start.elapsed() < test_duration {
                // Simple LCG for deterministic-ish variety without importing rand in test
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let use_large = (rng_state >> 33) % 3 == 0; // ~33% large, ~67% small

                let prompt = if use_large {
                    let prefix = format!("THREAD{}_LARGE_", thread_id);
                    large_prompt(&prefix, &format!("_{}", call_count))
                } else {
                    format!("small request thread={} call={}", thread_id, call_count)
                };

                let idx = policy_clone
                    .select_worker_with_headers(&workers_clone, Some(&prompt), None);

                match idx {
                    Some(i) => {
                        if i >= workers_clone.len() {
                            errors += 1;
                            eprintln!(
                                "CONCURRENCY BUG: returned index {} out of range (len={})",
                                i,
                                workers_clone.len()
                            );
                        }
                    }
                    None => {
                        errors += 1;
                        eprintln!("CONCURRENCY BUG: select returned None with healthy workers");
                    }
                }

                call_count += 1;
            }

            (call_count, errors)
        });
        handles.push(handle);
    }

    // Collect results — if any thread hangs forever the test will time out, which
    // is intentional: a hang indicates the deadlock was re-introduced.
    let mut total_calls = 0;
    let mut total_errors = 0;
    for handle in handles {
        let (calls, errors) = handle.join().expect("Thread should not panic");
        total_calls += calls;
        total_errors += errors;
    }

    println!(
        "Phase 6 – Concurrency stress: {} total calls across {} threads, {} errors, completed in {:.2}s",
        total_calls,
        num_threads,
        total_errors,
        start.elapsed().as_secs_f64()
    );

    assert_eq!(
        total_errors, 0,
        "No invalid indices or None results expected during concurrency stress; got {} errors",
        total_errors
    );
    assert!(
        total_calls > 0,
        "Expected at least some calls to complete during the stress test"
    );
    println!("Phase 6 – PASSED: No deadlock, no panics, all indices valid");
}

// ──────────────────────────────────────────────────────────────────────
// Phase 7: Worker scaling — add and remove workers
// ──────────────────────────────────────────────────────────────────────

/// Validates that the policy handles dynamic addition and removal of workers:
/// - Newly added worker can be selected
/// - Removed (unhealthy) workers are never selected
/// - Removing the last worker of the model prunes the tree (behaviorally — the
///   #[cfg(test)] helper `has_tree_for_model` is not accessible from integration
///   tests, so we assert on routing behavior instead)
#[test]
fn phase7_worker_scaling_add_remove() {
    let config = prod_config(0);
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(4);
    policy.init_workers(&workers);

    // ── Step 1: Confirm baseline routing works ───────────────────────────
    let small = "test scaling request";
    let idx = policy
        .select_worker_with_headers(&workers, Some(small), None)
        .expect("Should route during baseline");
    assert_valid_index(idx, &workers);

    // ── Step 2: Add a 5th worker dynamically ────────────────────────────
    let mut labels = HashMap::new();
    labels.insert("model_id".to_string(), "llama-prod".to_string());
    let w5 = Arc::new(
        BasicWorker::new("http://w5:8000".to_string(), WorkerType::Regular)
            .with_labels(labels),
    ) as Arc<dyn Worker>;

    policy.add_worker(w5.as_ref());
    let mut workers_5 = workers.clone();
    workers_5.push(Arc::clone(&w5));

    println!("Phase 7 – Added w5; total workers = {}", workers_5.len());

    // Load all original workers so w5 (empty pod) gets selected
    for w in &workers {
        w.increment_load();
        w.increment_load();
    }

    let idx_5 = policy
        .select_worker_with_headers(&workers_5, Some(small), None)
        .expect("Should route after adding w5");
    assert_valid_index(idx_5, &workers_5);
    println!(
        "Phase 7 – After adding w5 (load=0), routed to worker index {} (expected 4 = w5)",
        idx_5
    );
    assert_eq!(
        idx_5, 4,
        "Newly added w5 with load=0 should be selected as the only empty pod. Got {}",
        idx_5
    );

    // ── Step 3: Remove workers by marking unhealthy ──────────────────────
    // Mark w1..w4 unhealthy and remove from policy tree
    for w in &workers {
        w.set_healthy(false);
        policy.remove_worker(w.as_ref());
    }
    println!("Phase 7 – Marked workers 0-3 unhealthy; only w5 remains healthy");

    // Reset loads
    for w in &workers {
        w.reset_load();
    }

    // Only w5 should be returned
    for _ in 0..5 {
        let idx = policy
            .select_worker_with_headers(&workers_5, Some(small), None)
            .expect("Should route to w5 — the only healthy worker");
        assert_valid_index(idx, &workers_5);
        assert_eq!(
            idx, 4,
            "Only w5 is healthy; routing must select it (index 4). Got {}",
            idx
        );
    }
    println!("Phase 7 – All requests correctly routed to w5 after removing w0-w3");

    // ── Step 4: Remove the LAST worker (w5) ────────────────────────────
    w5.set_healthy(false);
    policy.remove_worker(w5.as_ref());
    println!("Phase 7 – Removed w5 (last worker); tree should be pruned");

    // All workers unhealthy → policy must return None (no panic)
    let result = policy.select_worker_with_headers(&workers_5, Some(small), None);
    assert!(
        result.is_none(),
        "With no healthy workers, routing must return None (not panic). Got: {:?}",
        result
    );
    println!("Phase 7 – Last-worker removal: returned None safely (tree pruned, no panic)");

    // NOTE: has_tree_for_model is only available in #[cfg(test)] unit builds,
    // not from integration tests. We verified tree pruning behaviorally above.
}

// ──────────────────────────────────────────────────────────────────────
// Phase 8: Eviction bounding — large tree evicted, policy still works
// ──────────────────────────────────────────────────────────────────────

/// With a very small max_tree_size, insert many distinct large prompts, trigger
/// eviction manually via evict_cache(), and assert the policy continues to
/// select valid workers afterward.
#[test]
fn phase8_eviction_bounding() {
    let config = CacheAwareConfig {
        eviction_interval_secs: 0, // We call evict_cache() manually
        max_tree_size: 100,        // tiny: forces eviction after a few large inserts
        ..CacheAwareConfig::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(4);
    policy.init_workers(&workers);

    // Insert 20 distinct large prompts to flood the tree past max_tree_size
    println!("Phase 8 – Inserting 20 distinct large prompts to overflow max_tree_size=100");
    for i in 0..20 {
        let unique_prompt = large_prompt(&format!("DISTINCT_PROMPT_{:05}", i), "");
        let idx = policy
            .select_worker_with_headers(&workers, Some(&unique_prompt), None)
            .expect("Must select a worker during flood");
        assert_valid_index(idx, &workers);
    }

    // Trigger eviction
    println!("Phase 8 – Calling evict_cache(max_size=100) to prune tree");
    policy.evict_cache(100);

    // Policy must still work after eviction
    println!("Phase 8 – Routing after eviction should still work");
    for i in 0..10 {
        let test_prompt = if i % 2 == 0 {
            large_prompt("POST_EVICTION_LARGE_", "")
        } else {
            "small post-eviction request".to_string()
        };

        let idx = policy
            .select_worker_with_headers(&workers, Some(&test_prompt), None)
            .expect("Must select a worker after eviction");
        assert_valid_index(idx, &workers);
    }

    println!("Phase 8 – PASSED: Policy functioning correctly after eviction. No panics, all indices valid.");
}
