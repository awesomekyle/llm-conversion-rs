//! Integration tests comparing our implementation to upstream llm-bridge
//!
//! These tests use test data generated from the TypeScript implementation
//! to verify that our Rust implementation matches the behavior.

use llm_conversion::{from_universal, to_universal, ProviderType, UniversalBody};
use serde_json::Value;
use std::fs;

fn load_test_data() -> Value {
    let data = fs::read_to_string("tests/integration_test_data.json")
        .expect("Failed to read integration test data");
    serde_json::from_str(&data).expect("Failed to parse test data")
}

fn compare_universal_bodies(rust: &UniversalBody, ts: &Value) {
    // Compare provider
    let ts_provider = ts.get("provider").and_then(|p| p.as_str()).unwrap();
    assert_eq!(rust.provider.to_string(), ts_provider);

    // Compare model
    let ts_model = ts.get("model").and_then(|m| m.as_str()).unwrap();
    assert_eq!(rust.model, ts_model);

    // Compare messages count
    let ts_messages = ts.get("messages").and_then(|m| m.as_array()).unwrap();
    assert_eq!(rust.messages.len(), ts_messages.len());

    // Compare system prompt existence
    let ts_has_system = ts.get("system").is_some();
    let rust_has_system = rust.system.is_some();
    assert_eq!(rust_has_system, ts_has_system);

    // Compare tools existence
    let ts_has_tools = ts.get("tools").is_some();
    let rust_has_tools = rust.tools.is_some();
    assert_eq!(rust_has_tools, ts_has_tools);
}

#[test]
fn test_integration_openai_basic() {
    let test_data = load_test_data();
    let test_case = test_data.get("openai_basic").unwrap();
    
    let input = test_case.get("input").unwrap().clone();
    let ts_universal = test_case.get("universal").unwrap();
    
    // Convert using Rust implementation
    let rust_universal = to_universal(ProviderType::OpenAI, input).unwrap();
    
    // Compare key fields
    compare_universal_bodies(&rust_universal, ts_universal);
}

#[test]
fn test_integration_openai_multimodal() {
    let test_data = load_test_data();
    let test_case = test_data.get("openai_multimodal").unwrap();
    
    let input = test_case.get("input").unwrap().clone();
    let ts_universal = test_case.get("universal").unwrap();
    
    let rust_universal = to_universal(ProviderType::OpenAI, input).unwrap();
    
    compare_universal_bodies(&rust_universal, ts_universal);
    
    // Verify multimodal content
    assert!(rust_universal.messages[0].content.len() > 1);
}

#[test]
fn test_integration_anthropic_basic() {
    let test_data = load_test_data();
    let test_case = test_data.get("anthropic_basic").unwrap();
    
    let input = test_case.get("input").unwrap().clone();
    let ts_universal = test_case.get("universal").unwrap();
    
    let rust_universal = to_universal(ProviderType::Anthropic, input).unwrap();
    
    compare_universal_bodies(&rust_universal, ts_universal);
}

#[test]
fn test_integration_anthropic_multimodal() {
    let test_data = load_test_data();
    let test_case = test_data.get("anthropic_multimodal").unwrap();
    
    let input = test_case.get("input").unwrap().clone();
    let ts_universal = test_case.get("universal").unwrap();
    
    let rust_universal = to_universal(ProviderType::Anthropic, input).unwrap();
    
    compare_universal_bodies(&rust_universal, ts_universal);
    
    // Verify multimodal content
    assert!(rust_universal.messages[0].content.len() > 1);
}

#[test]
fn test_integration_google_basic() {
    let test_data = load_test_data();
    let test_case = test_data.get("google_basic").unwrap();
    
    let input = test_case.get("input").unwrap().clone();
    
    let rust_universal = to_universal(ProviderType::Google, input).unwrap();
    
    // Basic comparisons for Google
    assert_eq!(rust_universal.provider, ProviderType::Google);
    assert!(!rust_universal.messages.is_empty());
}

#[test]
fn test_integration_google_multimodal() {
    let test_data = load_test_data();
    let test_case = test_data.get("google_multimodal").unwrap();
    
    let input = test_case.get("input").unwrap().clone();
    
    let rust_universal = to_universal(ProviderType::Google, input).unwrap();
    
    assert_eq!(rust_universal.provider, ProviderType::Google);
    assert!(rust_universal.messages[0].content.len() > 1);
}

#[test]
fn test_roundtrip_all_providers() {
    let test_data = load_test_data();
    
    for provider_name in ["openai", "anthropic", "google"] {
        let test_name = format!("{}_basic", provider_name);
        let test_case = test_data.get(&test_name);
        
        if test_case.is_none() {
            continue;
        }
        
        let test_case = test_case.unwrap();
        let input = test_case.get("input").unwrap().clone();
        
        let provider = match provider_name {
            "openai" => ProviderType::OpenAI,
            "anthropic" => ProviderType::Anthropic,
            "google" => ProviderType::Google,
            _ => continue,
        };
        
        // Roundtrip: input -> universal -> output
        let universal = to_universal(provider, input.clone()).unwrap();
        let output = from_universal(provider, &universal).unwrap();
        
        // Basic sanity checks
        match provider {
            ProviderType::OpenAI => {
                assert!(output.get("messages").is_some());
                assert!(output.get("model").is_some());
            }
            ProviderType::Anthropic => {
                assert!(output.get("messages").is_some());
                assert!(output.get("model").is_some());
                assert!(output.get("max_tokens").is_some());
            }
            ProviderType::Google => {
                assert!(output.get("contents").is_some());
            }
        }
    }
}
