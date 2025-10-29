//! # LLM Conversion
//!
//! Rust implementation of llm-bridge - Universal Translation Layer for Large Language Model APIs.
//!
//! This library provides seamless translation between different LLM provider APIs (OpenAI, Anthropic, Google)
//! while preserving zero data loss and enabling perfect reconstruction of original requests.
//!
//! ## Features
//!
//! - **Perfect Translation** - Convert between OpenAI, Anthropic, and Google formats
//! - **Zero Data Loss** - Every field is preserved with `_original` reconstruction
//! - **Multimodal Support** - Images, documents, and rich content across providers
//! - **Tool Calling** - Function calling translation between different formats
//! - **Type Safety** - Full Rust type safety with serde support
//!
//! ## Example
//!
//! ```rust
//! use llm_conversion::{to_universal, from_universal, ProviderType};
//! use serde_json::json;
//!
//! // Convert OpenAI request to universal format
//! let openai_request = json!({
//!     "model": "gpt-4",
//!     "messages": [
//!         {"role": "user", "content": "Hello!"}
//!     ]
//! });
//!
//! let universal = to_universal(ProviderType::OpenAI, openai_request).unwrap();
//!
//! // Convert to Anthropic format
//! let anthropic_request = from_universal(ProviderType::Anthropic, &universal).unwrap();
//! ```

mod anthropic;
mod converter;
mod error;
mod google;
mod openai;
mod types;
mod utils;

// Re-export public API
pub use converter::{detect_provider, from_universal, to_universal, translate_between_providers};
pub use error::{ConversionError, Result};
pub use types::{
    ContentType, ProviderType, ToolChoice, UniversalBody, UniversalContent, UniversalMediaContent,
    UniversalMessage, UniversalRole, UniversalSystemPrompt, UniversalTool, UniversalToolCall,
    UniversalToolResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_workflow() {
        let openai_body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        });

        // Convert to universal
        let universal = to_universal(ProviderType::OpenAI, openai_body.clone()).unwrap();
        assert_eq!(universal.provider, ProviderType::OpenAI);
        assert_eq!(universal.model, "gpt-4");

        // Convert back to OpenAI
        let reconstructed = from_universal(ProviderType::OpenAI, &universal).unwrap();
        assert_eq!(reconstructed.get("model"), openai_body.get("model"));

        // Translate to Anthropic
        let anthropic = from_universal(ProviderType::Anthropic, &universal).unwrap();
        assert!(anthropic.get("messages").is_some());
        assert!(anthropic.get("max_tokens").is_some());
    }
}
