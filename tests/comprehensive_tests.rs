//! Comprehensive tests for all conversion functionality

use llm_conversion::{from_universal, to_universal, translate_between_providers, ProviderType, ContentType};
use serde_json::json;

#[test]
fn test_openai_tool_calls() {
    let openai_body = json!({
        "model": "gpt-4",
        "messages": [
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"San Francisco\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "content": "{\"temperature\": 72}",
                "tool_call_id": "call_123"
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }]
    });

    let universal = to_universal(ProviderType::OpenAI, openai_body).unwrap();
    assert!(universal.tools.is_some());
    assert_eq!(universal.messages.len(), 2);
    assert!(universal.messages[0].tool_calls.is_some());
}

#[test]
fn test_anthropic_tool_use() {
    let anthropic_body = json!({
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }
            ]
        }],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }]
    });

    let universal = to_universal(ProviderType::Anthropic, anthropic_body).unwrap();
    assert!(universal.tools.is_some());
    assert_eq!(universal.messages[0].content.len(), 2);
}

#[test]
fn test_google_function_calling() {
    let google_body = json!({
        "contents": [{
            "role": "model",
            "parts": [{
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "San Francisco"}
                }
            }]
        }],
        "tools": [{
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }]
        }]
    });

    let universal = to_universal(ProviderType::Google, google_body).unwrap();
    assert!(universal.tools.is_some());
    assert_eq!(universal.messages[0].content.len(), 1);
}

#[test]
fn test_cross_provider_translation() {
    let openai_body = json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7
    });

    // OpenAI -> Anthropic
    let anthropic = translate_between_providers(
        ProviderType::OpenAI,
        ProviderType::Anthropic,
        openai_body.clone(),
    )
    .unwrap();
    assert!(anthropic.get("max_tokens").is_some());
    assert!(anthropic.get("system").is_some());

    // OpenAI -> Google
    let google = translate_between_providers(
        ProviderType::OpenAI,
        ProviderType::Google,
        openai_body,
    )
    .unwrap();
    assert!(google.get("contents").is_some());
    assert!(google.get("systemInstruction").is_some());
}

#[test]
fn test_multimodal_cross_provider() {
    let openai_multimodal = json!({
        "model": "gpt-4-vision",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,abc123",
                        "detail": "high"
                    }
                }
            ]
        }]
    });

    let universal = to_universal(ProviderType::OpenAI, openai_multimodal).unwrap();
    
    // Convert to Anthropic
    let anthropic = from_universal(ProviderType::Anthropic, &universal).unwrap();
    let msgs = anthropic.get("messages").unwrap().as_array().unwrap();
    let content = msgs[0].get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 2);
    
    // Verify image block has proper structure for Anthropic
    let image_block = &content[1];
    assert_eq!(image_block.get("type").unwrap(), "image");
    assert!(image_block.get("source").is_some());
}

#[test]
fn test_roundtrip_preservation() {
    let original = json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    });

    let universal = to_universal(ProviderType::OpenAI, original.clone()).unwrap();
    let reconstructed = from_universal(ProviderType::OpenAI, &universal).unwrap();

    assert_eq!(original.get("model"), reconstructed.get("model"));
    assert_eq!(original.get("max_tokens"), reconstructed.get("max_tokens"));
    // Check temperature with tolerance for floating point
    let orig_temp = original.get("temperature").unwrap().as_f64().unwrap();
    let recon_temp = reconstructed.get("temperature").unwrap().as_f64().unwrap();
    assert!((orig_temp - recon_temp).abs() < 0.01);
}

#[test]
fn test_anthropic_cache_control() {
    let anthropic_body = json!({
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }]
    });

    let universal = to_universal(ProviderType::Anthropic, anthropic_body).unwrap();
    assert!(universal.messages[0].metadata.cache_control.is_some());
}

#[test]
fn test_google_file_data() {
    let google_body = json!({
        "contents": [{
            "role": "user",
            "parts": [{
                "fileData": {
                    "mimeType": "application/pdf",
                    "fileUri": "gs://bucket/file.pdf"
                }
            }]
        }]
    });

    let universal = to_universal(ProviderType::Google, google_body).unwrap();
    assert_eq!(universal.messages[0].content[0].content_type, ContentType::Document);
    
    let media = universal.messages[0].content[0].media.as_ref().unwrap();
    assert_eq!(media.file_uri, Some("gs://bucket/file.pdf".to_string()));
}

#[test]
fn test_all_parameters() {
    let openai_body = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Test"}],
        "temperature": 0.5,
        "max_tokens": 100,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "seed": 42
    });

    let universal = to_universal(ProviderType::OpenAI, openai_body).unwrap();
    assert_eq!(universal.temperature, Some(0.5));
    assert_eq!(universal.max_tokens, Some(100));
    assert_eq!(universal.top_p, Some(0.9));
    assert_eq!(universal.frequency_penalty, Some(0.5));
    assert_eq!(universal.presence_penalty, Some(0.3));
    assert_eq!(universal.seed, Some(42));
}
