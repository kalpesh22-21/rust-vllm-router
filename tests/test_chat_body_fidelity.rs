//! Integration test for the raw-body-forwarding guarantee introduced for
//! /v1/chat/completions.
//!
//! The regular HTTP Router (`crate::routers::http::router::Router`) now accepts
//! an optional `raw_body: Option<Bytes>` parameter on `route_chat`.  When the
//! bytes are present they are forwarded byte-for-byte to the upstream worker
//! instead of serialising the parsed `ChatCompletionRequest` struct.
//!
//! This file verifies the **core fidelity guarantee**: the body the upstream
//! worker receives is byte-identical to the bytes that were passed as
//! `raw_body`, and in particular does NOT contain fields that would have been
//! injected by struct serialisation (e.g. `separate_reasoning`,
//! `stream_reasoning`, `add_generation_prompt`, `skip_special_tokens`).

mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    response::IntoResponse,
    routing::post,
    Router as AxumRouter,
};
use bytes::Bytes;
// common::mock_worker is pulled in by the `mod common;` declaration above and
// needed for the module tree, but we don't use its types directly here.
use reqwest::Client;
use serde_json::json;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{
        CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    },
    protocols::spec::ChatCompletionRequest,
    routers::{RouterFactory, RouterTrait},
};

// ---------------------------------------------------------------------------
// A minimal body-echoing mock upstream
// ---------------------------------------------------------------------------

/// State shared between the echo server and the test assertion.
#[derive(Clone, Default)]
struct EchoState {
    /// Raw bytes received by the last POST to /v1/chat/completions
    captured_body: Arc<Mutex<Option<Bytes>>>,
}

/// Handler that stores the raw request body for later inspection and returns
/// a minimal valid chat-completion JSON so the router doesn't error out.
async fn echo_chat_handler(
    axum::extract::State(state): axum::extract::State<EchoState>,
    req: Request<Body>,
) -> impl IntoResponse {
    let body_bytes = axum::body::to_bytes(req.into_body(), usize::MAX)
        .await
        .unwrap_or_default();

    *state.captured_body.lock().unwrap() = Some(body_bytes);

    // Return a minimal but structurally valid non-streaming chat response so
    // the router pipeline doesn't reject it.
    let response_json = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });

    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&response_json).unwrap(),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Helper: build a Regular Router pointing at a given worker URL
// ---------------------------------------------------------------------------

async fn build_regular_router(worker_url: String) -> Arc<dyn RouterTrait> {
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url],
        },
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3099,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
        metrics: None,
        log_dir: None,
        log_level: None,
        request_id_headers: None,
        max_concurrent_requests: 64,
        queue_size: 0,
        queue_timeout_secs: 60,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        model_path: None,
        tokenizer_path: None,
        history_backend: vllm_router_rs::config::HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 30,
    };

    let app_context = common::create_test_context(config);
    let router = RouterFactory::create_router(&app_context).await.unwrap();
    Arc::from(router)
}

// ---------------------------------------------------------------------------
// Test: raw bytes are forwarded byte-for-byte
// ---------------------------------------------------------------------------

/// Start the echo server as a real TCP listener (same pattern as
/// `MockOpenAIServer`) so the regular Router can hit it over HTTP.
async fn start_echo_server() -> (String, Arc<Mutex<Option<Bytes>>>) {
    let captured_body: Arc<Mutex<Option<Bytes>>> = Arc::new(Mutex::new(None));

    let state = EchoState {
        captured_body: Arc::clone(&captured_body),
    };

    let app = AxumRouter::new()
        // The regular Router also GETs /health during startup checks.
        .route(
            "/health",
            axum::routing::get(|| async { StatusCode::OK.into_response() }),
        )
        // Also required for worker discovery: /get_server_info
        .route(
            "/get_server_info",
            axum::routing::get(|| async {
                (
                    StatusCode::OK,
                    [(CONTENT_TYPE, "application/json")],
                    serde_json::to_string(&json!({
                        "model_path": "test-model",
                        "tokenizer_path": "test-tokenizer",
                        "port": 0,
                        "host": "127.0.0.1",
                        "context_length": 4096,
                        "version": "0.0.0"
                    }))
                    .unwrap(),
                )
                    .into_response()
            }),
        )
        .route("/v1/chat/completions", post(echo_chat_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give the server a moment to start accepting connections.
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    (url, captured_body)
}

/// Core fidelity test: bytes passed as `raw_body` must arrive at the upstream
/// worker unchanged — no injected booleans or extra keys from struct
/// serialisation.
#[tokio::test]
async fn test_chat_raw_body_forwarded_byte_for_byte() {
    // 1. Start the echo upstream.
    let (worker_url, captured_body) = start_echo_server().await;

    // 2. Build the regular Router (same pattern as api_endpoints_test.rs).
    let router = build_regular_router(worker_url).await;

    // Wait a moment for the router's background worker-health monitor to
    // register the worker as healthy.
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // 3. Craft a minimal, known JSON payload.
    //    It intentionally omits fields that struct-serialisation would inject
    //    (e.g. `separate_reasoning`, `stream_reasoning`, `skip_special_tokens`,
    //    `add_generation_prompt`).
    let raw_bytes = Bytes::from_static(
        br#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
    );

    // 4. Parse the same bytes into the typed struct (mirrors server.rs).
    let parsed: ChatCompletionRequest = serde_json::from_slice(&raw_bytes).unwrap();

    // 5. Route through the regular Router passing raw_body.
    let response = router
        .route_chat(None, &parsed, None, Some(raw_bytes.clone()))
        .await;

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Router should forward the request successfully"
    );

    // 6. Assert the upstream received exactly the bytes we sent.
    let received = captured_body
        .lock()
        .unwrap()
        .clone()
        .expect("Echo server should have received a request body");

    assert_eq!(
        received, raw_bytes,
        "Upstream must receive the raw bytes byte-for-byte; \
         serialisation of the parsed struct must NOT be used"
    );

    // 7. Extra sanity: the received bytes must NOT contain any field that would
    //    have been injected by serialising ChatCompletionRequest.
    let received_str = std::str::from_utf8(&received).unwrap();
    for injected_field in &[
        "separate_reasoning",
        "stream_reasoning",
        "add_generation_prompt",
        "skip_special_tokens",
    ] {
        assert!(
            !received_str.contains(injected_field),
            "Injected field '{}' must not appear in the forwarded body; got: {}",
            injected_field,
            received_str
        );
    }
}

// ---------------------------------------------------------------------------
// Test: when raw_body is None the router falls back to struct serialisation
// (regression guard — ensures the None path still works)
// ---------------------------------------------------------------------------

/// When `raw_body` is `None` the router serialises the typed struct and sends
/// it. This is the previous behaviour and must continue to work.
#[tokio::test]
async fn test_chat_no_raw_body_falls_back_to_struct_serialisation() {
    let (worker_url, captured_body) = start_echo_server().await;
    let router = build_regular_router(worker_url).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let parsed: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();

    let response = router
        .route_chat(None, &parsed, None, None)
        .await;

    assert_eq!(response.status(), StatusCode::OK);

    // The upstream should still have received valid JSON.
    let received = captured_body
        .lock()
        .unwrap()
        .clone()
        .expect("Echo server should have received a request body");

    let received_value: serde_json::Value = serde_json::from_slice(&received)
        .expect("Upstream body must be valid JSON even without raw_body");

    assert_eq!(
        received_value.get("model").and_then(|v| v.as_str()),
        Some("test"),
        "Model field must survive struct serialisation round-trip"
    );
}

// ---------------------------------------------------------------------------
// Test: end-to-end through the full Axum server stack
// ---------------------------------------------------------------------------

/// The full `/v1/chat/completions` handler in server.rs reads raw bytes,
/// parses them, and passes both to `route_chat`. This test exercises that
/// path through `create_test_app` (same pattern as api_endpoints_test.rs)
/// and verifies the echo server sees the original bytes.
#[tokio::test]
async fn test_chat_e2e_raw_body_forwarded_through_server_stack() {
    // Start a normal mock worker (just needs /health for router startup).
    // We also start our echo server to capture the body.
    let (echo_url, captured_body) = start_echo_server().await;

    let config = vllm_router_rs::config::RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![echo_url],
        },
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3098,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
        metrics: None,
        log_dir: None,
        log_level: None,
        request_id_headers: None,
        max_concurrent_requests: 64,
        queue_size: 0,
        queue_timeout_secs: 60,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        model_path: None,
        tokenizer_path: None,
        history_backend: vllm_router_rs::config::HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 30,
    };

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .unwrap();

    let app_context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&app_context).await.unwrap();
    let router = Arc::from(router);

    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let app = common::test_app::create_test_app(Arc::clone(&router), client, &config);

    // Send a request with a known exact body through the full server stack.
    let raw_body = br#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#;

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(raw_body.as_ref()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let received = captured_body
        .lock()
        .unwrap()
        .clone()
        .expect("Echo server must have received a body");

    assert_eq!(
        received.as_ref(),
        raw_body,
        "Full-stack path: upstream body must be byte-identical to the original client body"
    );
}
