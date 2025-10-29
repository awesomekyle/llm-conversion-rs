use crate::error::{ConversionError, Result};
use crate::types::{ProviderType, UniversalBody};
use serde_json::Value;

/// Detect provider type from request body
pub fn detect_provider(body: &Value) -> Result<ProviderType> {
    if !body.is_object() {
        return Err(ConversionError::DetectionFailed(
            "Request body must be an object".to_string(),
        ));
    }

    let obj = body.as_object().unwrap();

    // Check for Google-specific fields
    if obj.contains_key("contents") {
        return Ok(ProviderType::Google);
    }

    // Check for Anthropic-specific fields
    if obj.contains_key("max_tokens") && !obj.contains_key("max_completion_tokens") {
        // Anthropic requires max_tokens, OpenAI uses max_tokens or max_completion_tokens optionally
        if obj.get("messages").and_then(|m| m.as_array()).is_some() {
            // Check if messages have Anthropic-style content
            if let Some(messages) = obj.get("messages").and_then(|m| m.as_array()) {
                if let Some(first_msg) = messages.first() {
                    if let Some(role) = first_msg.get("role").and_then(|r| r.as_str()) {
                        // Anthropic only supports "user" and "assistant" roles
                        // OpenAI supports "system", "user", "assistant", "tool"
                        if role == "user" || role == "assistant" {
                            // Could be either, check for more Anthropic-specific markers
                            if obj.contains_key("anthropic_version") {
                                return Ok(ProviderType::Anthropic);
                            }
                            // Check message content structure
                            if let Some(content) = first_msg.get("content") {
                                if let Some(arr) = content.as_array() {
                                    if let Some(first_content) = arr.first() {
                                        if first_content.get("type").and_then(|t| t.as_str())
                                            == Some("tool_use")
                                            || first_content.get("type").and_then(|t| t.as_str())
                                                == Some("tool_result")
                                        {
                                            return Ok(ProviderType::Anthropic);
                                        }
                                    }
                                }
                            }
                            // Default to Anthropic if max_tokens is required
                            return Ok(ProviderType::Anthropic);
                        }
                    }
                }
            }
        }
    }

    // Check for OpenAI-specific fields
    if obj.contains_key("messages") {
        // Check for system role or tool role
        if let Some(messages) = obj.get("messages").and_then(|m| m.as_array()) {
            for msg in messages {
                if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                    if role == "system" || role == "tool" || role == "developer" {
                        return Ok(ProviderType::OpenAI);
                    }
                }
                // Check for OpenAI-specific tool_calls
                if msg.get("tool_calls").is_some() {
                    return Ok(ProviderType::OpenAI);
                }
                if msg.get("tool_call_id").is_some() {
                    return Ok(ProviderType::OpenAI);
                }
            }
        }
        // Default to OpenAI if it has messages
        return Ok(ProviderType::OpenAI);
    }

    Err(ConversionError::DetectionFailed(
        "Could not determine provider from request body".to_string(),
    ))
}

/// Convert provider-specific format to Universal format
pub fn to_universal(provider: ProviderType, body: Value) -> Result<UniversalBody> {
    match provider {
        ProviderType::OpenAI => crate::openai::openai_to_universal(body),
        ProviderType::Anthropic => crate::anthropic::anthropic_to_universal(body),
        ProviderType::Google => crate::google::google_to_universal(body),
    }
}

/// Convert Universal format to provider-specific format
pub fn from_universal(provider: ProviderType, universal: &UniversalBody) -> Result<Value> {
    match provider {
        ProviderType::OpenAI => crate::openai::universal_to_openai(universal),
        ProviderType::Anthropic => crate::anthropic::universal_to_anthropic(universal),
        ProviderType::Google => crate::google::universal_to_google(universal),
    }
}

/// Translate between two providers directly
pub fn translate_between_providers(
    from_provider: ProviderType,
    to_provider: ProviderType,
    body: Value,
) -> Result<Value> {
    // Convert to universal format
    let universal = to_universal(from_provider, body)?;

    // Update provider in universal
    let mut updated_universal = universal;
    updated_universal.provider = to_provider;

    // Convert to target provider format
    from_universal(to_provider, &updated_universal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_detect_openai() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ]
        });

        assert_eq!(detect_provider(&body).unwrap(), ProviderType::OpenAI);
    }

    #[test]
    fn test_detect_anthropic() {
        let body = json!({
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        assert_eq!(detect_provider(&body).unwrap(), ProviderType::Anthropic);
    }

    #[test]
    fn test_detect_google() {
        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello"}]
            }]
        });

        assert_eq!(detect_provider(&body).unwrap(), ProviderType::Google);
    }

    #[test]
    fn test_translate_openai_to_anthropic() {
        let openai_body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7
        });

        let result =
            translate_between_providers(ProviderType::OpenAI, ProviderType::Anthropic, openai_body);

        assert!(result.is_ok());
        let anthropic_body = result.unwrap();
        assert_eq!(
            anthropic_body.get("model").and_then(|m| m.as_str()),
            Some("gpt-4")
        );
        assert!(anthropic_body.get("messages").is_some());
        assert!(anthropic_body.get("max_tokens").is_some());
    }
}
