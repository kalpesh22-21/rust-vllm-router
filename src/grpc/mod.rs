//! gRPC client module for communicating with VLLM scheduler
//!
//! This module provides a gRPC client implementation for the VLLM router.

pub mod client;

// Re-export the client
pub use client::{proto, VllmSchedulerClient};
