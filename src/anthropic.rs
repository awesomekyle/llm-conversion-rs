use crate::error::{ConversionError, Result};
use crate::types::*;
use crate::utils::generate_id;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Convert Anthropic format to Universal format
pub fn anthropic_to_universal(body: Value) -> Result<UniversalBody> {
    if !body.is_object() {
        return Err(ConversionError::InvalidFormat(
            "Request body must be an object".to_string(),
        ));
    }

    let obj = body.as_object().unwrap();

    // Extract messages
    let messages = obj
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ConversionError::MissingField("messages".to_string()))?;

    // Convert messages
    let universal_messages: Result<Vec<UniversalMessage>> = messages
        .iter()
        .enumerate()
        .map(|(index, msg)| parse_anthropic_message(msg, index))
        .collect();

    // Extract system prompt
    let system = obj.get("system").map(|s| {
        if let Some(str_val) = s.as_str() {
            UniversalSystemPrompt::String(str_val.to_string())
        } else if let Some(arr) = s.as_array() {
            // Complex system prompt with multiple parts
            let content = arr
                .iter()
                .filter_map(|part| {
                    if part.get("type")?.as_str()? == "text" {
                        part.get("text")?.as_str().map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            UniversalSystemPrompt::String(content)
        } else {
            UniversalSystemPrompt::String(s.to_string())
        }
    });

    // Extract tools
    let tools = obj
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools_array| {
            tools_array
                .iter()
                .filter_map(|tool| parse_anthropic_tool(tool).ok())
                .collect()
        });

    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(UniversalBody {
        provider: ProviderType::Anthropic,
        system,
        messages: universal_messages?,
        model,
        temperature: obj
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        max_tokens: obj
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        top_p: obj.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        stream: obj.get("stream").and_then(|v| v.as_bool()),
        tools,
        tool_choice: obj.get("tool_choice").map(parse_anthropic_tool_choice),
        provider_params: extract_anthropic_params(obj),
        _original: Some(OriginalContent {
            provider: ProviderType::Anthropic,
            raw: body,
        }),
    })
}

/// Convert Universal format to Anthropic format
pub fn universal_to_anthropic(universal: &UniversalBody) -> Result<Value> {
    let mut result = json!({
        "model": universal.model,
        "max_tokens": universal.max_tokens.unwrap_or(1024)
    });

    let obj = result.as_object_mut().unwrap();

    // Add system prompt if present
    if let Some(system) = &universal.system {
        let system_content = match system {
            UniversalSystemPrompt::String(s) => json!(s),
            UniversalSystemPrompt::Complex { content, .. } => json!(content),
        };
        obj.insert("system".to_string(), system_content);
    }

    // Convert messages
    let anthropic_messages: Vec<Value> = universal
        .messages
        .iter()
        .filter_map(|msg| convert_universal_message_to_anthropic(msg).ok())
        .collect();

    obj.insert("messages".to_string(), json!(anthropic_messages));

    // Add optional parameters
    if let Some(temp) = universal.temperature {
        obj.insert("temperature".to_string(), json!(temp));
    }
    if let Some(top_p) = universal.top_p {
        obj.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(stream) = universal.stream {
        obj.insert("stream".to_string(), json!(stream));
    }

    // Add tools if present
    if let Some(tools) = &universal.tools {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .map(convert_universal_tool_to_anthropic)
            .collect();
        obj.insert("tools".to_string(), json!(anthropic_tools));
    }

    // Add tool_choice if present
    if let Some(tool_choice) = &universal.tool_choice {
        obj.insert(
            "tool_choice".to_string(),
            convert_tool_choice_to_anthropic(tool_choice),
        );
    }

    // Add provider-specific params
    if let Some(params) = &universal.provider_params {
        for (key, value) in params {
            obj.insert(key.clone(), value.clone());
        }
    }

    Ok(result)
}

fn parse_anthropic_message(msg: &Value, index: usize) -> Result<UniversalMessage> {
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .ok_or_else(|| ConversionError::MissingField("role".to_string()))?;

    let universal_role = match role {
        "user" => UniversalRole::User,
        "assistant" => UniversalRole::Assistant,
        _ => UniversalRole::User,
    };

    let content = parse_anthropic_content(msg.get("content"))?;

    // Check for cache control
    let cache_control = if let Some(content_arr) = msg.get("content").and_then(|c| c.as_array()) {
        content_arr.iter().find_map(|block| {
            block
                .get("cache_control")
                .and_then(|cc| cc.as_object().cloned())
                .map(|map| map.into_iter().collect::<HashMap<String, Value>>())
        })
    } else {
        None
    };

    Ok(UniversalMessage {
        id: generate_id(),
        role: universal_role,
        content,
        metadata: MessageMetadata {
            provider: ProviderType::Anthropic,
            original_role: None,
            original_index: Some(index),
            cache_control,
            name: None,
            tool_call_id: None,
            parts_metadata: None,
            extra: HashMap::new(),
        },
        tool_calls: None,
    })
}

fn parse_anthropic_content(content: Option<&Value>) -> Result<Vec<UniversalContent>> {
    match content {
        None => Ok(vec![]),
        Some(Value::String(s)) => Ok(vec![UniversalContent {
            content_type: ContentType::Text,
            text: Some(s.clone()),
            media: None,
            tool_call: None,
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::Anthropic,
                raw: json!(s),
            }),
        }]),
        Some(Value::Array(arr)) => {
            let contents: Vec<UniversalContent> = arr
                .iter()
                .filter_map(|block| parse_anthropic_content_block(block).ok())
                .collect();
            Ok(contents)
        }
        Some(other) => Ok(vec![UniversalContent {
            content_type: ContentType::Text,
            text: Some(serde_json::to_string(other).unwrap_or_default()),
            media: None,
            tool_call: None,
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::Anthropic,
                raw: other.clone(),
            }),
        }]),
    }
}

fn parse_anthropic_content_block(block: &Value) -> Result<UniversalContent> {
    let block_type = block
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Content block missing type".to_string()))?;

    match block_type {
        "text" => {
            let text = block.get("text").and_then(|t| t.as_str()).ok_or_else(|| {
                ConversionError::InvalidFormat("Text block missing text".to_string())
            })?;
            Ok(UniversalContent {
                content_type: ContentType::Text,
                text: Some(text.to_string()),
                media: None,
                tool_call: None,
                tool_result: None,
                _original: Some(OriginalContent {
                    provider: ProviderType::Anthropic,
                    raw: block.clone(),
                }),
            })
        }
        "image" => {
            let source = block.get("source").ok_or_else(|| {
                ConversionError::InvalidFormat("Image block missing source".to_string())
            })?;

            let source_type = source
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("base64");

            let media = if source_type == "base64" {
                UniversalMediaContent {
                    url: None,
                    detail: None,
                    data: source
                        .get("data")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    mime_type: source
                        .get("media_type")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string()),
                    file_uri: None,
                    file_name: None,
                    size: None,
                    duration: None,
                    metadata: None,
                }
            } else {
                UniversalMediaContent {
                    url: source
                        .get("url")
                        .and_then(|u| u.as_str())
                        .map(|s| s.to_string()),
                    detail: None,
                    data: None,
                    mime_type: None,
                    file_uri: None,
                    file_name: None,
                    size: None,
                    duration: None,
                    metadata: None,
                }
            };

            Ok(UniversalContent {
                content_type: ContentType::Image,
                text: None,
                media: Some(media),
                tool_call: None,
                tool_result: None,
                _original: Some(OriginalContent {
                    provider: ProviderType::Anthropic,
                    raw: block.clone(),
                }),
            })
        }
        "tool_use" => {
            let id = block
                .get("id")
                .and_then(|i| i.as_str())
                .ok_or_else(|| ConversionError::InvalidFormat("Tool use missing id".to_string()))?;
            let name = block.get("name").and_then(|n| n.as_str()).ok_or_else(|| {
                ConversionError::InvalidFormat("Tool use missing name".to_string())
            })?;
            let input = block.get("input").cloned().unwrap_or(json!({}));

            let arguments: HashMap<String, Value> = if let Some(obj) = input.as_object() {
                obj.clone().into_iter().collect()
            } else {
                HashMap::new()
            };

            let mut metadata = HashMap::new();
            metadata.insert("input".to_string(), input);

            Ok(UniversalContent {
                content_type: ContentType::ToolCall,
                text: None,
                media: None,
                tool_call: Some(UniversalToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments,
                    metadata: Some(metadata),
                }),
                tool_result: None,
                _original: Some(OriginalContent {
                    provider: ProviderType::Anthropic,
                    raw: block.clone(),
                }),
            })
        }
        "tool_result" => {
            let tool_use_id = block
                .get("tool_use_id")
                .and_then(|i| i.as_str())
                .ok_or_else(|| {
                    ConversionError::InvalidFormat("Tool result missing tool_use_id".to_string())
                })?;

            let content = block.get("content").cloned().unwrap_or(json!(null));

            let mut metadata = HashMap::new();
            metadata.insert("tool_use_id".to_string(), json!(tool_use_id));
            metadata.insert("content".to_string(), content.clone());

            Ok(UniversalContent {
                content_type: ContentType::ToolResult,
                text: None,
                media: None,
                tool_call: None,
                tool_result: Some(UniversalToolResult {
                    tool_call_id: tool_use_id.to_string(),
                    name: String::new(),
                    result: content,
                    error: None,
                    metadata: Some(metadata),
                }),
                _original: Some(OriginalContent {
                    provider: ProviderType::Anthropic,
                    raw: block.clone(),
                }),
            })
        }
        _ => Err(ConversionError::InvalidFormat(format!(
            "Unknown content block type: {}",
            block_type
        ))),
    }
}

fn parse_anthropic_tool(tool: &Value) -> Result<UniversalTool> {
    let name = tool
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Tool missing name".to_string()))?;

    let description = tool
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("");

    let input_schema = tool
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let mut metadata = HashMap::new();
    metadata.insert("input_schema".to_string(), input_schema.clone());

    Ok(UniversalTool {
        name: name.to_string(),
        description: description.to_string(),
        parameters: input_schema,
        metadata: Some(metadata),
        _original: Some(OriginalContent {
            provider: ProviderType::Anthropic,
            raw: tool.clone(),
        }),
    })
}

fn parse_anthropic_tool_choice(tc: &Value) -> ToolChoice {
    if let Some(obj) = tc.as_object() {
        if let Some(tc_type) = obj.get("type").and_then(|t| t.as_str()) {
            match tc_type {
                "auto" => ToolChoice::String("auto".to_string()),
                "any" => ToolChoice::String("required".to_string()),
                "tool" => {
                    if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                        ToolChoice::Named {
                            name: name.to_string(),
                        }
                    } else {
                        ToolChoice::String("auto".to_string())
                    }
                }
                _ => ToolChoice::String("auto".to_string()),
            }
        } else {
            ToolChoice::String("auto".to_string())
        }
    } else {
        ToolChoice::String("auto".to_string())
    }
}

fn extract_anthropic_params(
    obj: &serde_json::Map<String, Value>,
) -> Option<HashMap<String, Value>> {
    let mut params = HashMap::new();

    if let Some(anthropic_version) = obj.get("anthropic_version") {
        params.insert("anthropic_version".to_string(), anthropic_version.clone());
    }
    if let Some(stop_sequences) = obj.get("stop_sequences") {
        params.insert("stop_sequences".to_string(), stop_sequences.clone());
    }
    if let Some(metadata) = obj.get("metadata") {
        params.insert("metadata".to_string(), metadata.clone());
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

fn convert_universal_message_to_anthropic(msg: &UniversalMessage) -> Result<Value> {
    let role = match msg.role {
        UniversalRole::User => "user",
        UniversalRole::Assistant => "assistant",
        _ => "user", // Anthropic only supports user and assistant
    };

    // Convert content blocks
    let content_blocks: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|c| convert_universal_content_to_anthropic(c).ok())
        .collect();

    // If message is assistant with tool calls in content, keep them as tool_use blocks
    let has_tool_calls = msg
        .content
        .iter()
        .any(|c| c.content_type == ContentType::ToolCall);

    let content = if content_blocks.len() == 1
        && !has_tool_calls
        && content_blocks[0].get("type").and_then(|t| t.as_str()) == Some("text")
    {
        // Simple text message
        content_blocks[0].get("text").cloned().unwrap_or(json!(""))
    } else {
        // Complex content
        json!(content_blocks)
    };

    Ok(json!({
        "role": role,
        "content": content
    }))
}

fn convert_universal_content_to_anthropic(content: &UniversalContent) -> Result<Value> {
    match content.content_type {
        ContentType::Text => Ok(json!({
            "type": "text",
            "text": content.text.as_ref().unwrap_or(&String::new())
        })),
        ContentType::Image => {
            if let Some(media) = &content.media {
                let source = if let (Some(data), Some(mime_type)) = (&media.data, &media.mime_type)
                {
                    json!({
                        "type": "base64",
                        "media_type": mime_type,
                        "data": data
                    })
                } else if let Some(url) = &media.url {
                    // Extract from data URL if present
                    if url.starts_with("data:") {
                        if let Some(caps) = url.strip_prefix("data:") {
                            if let Some(semicolon_pos) = caps.find(';') {
                                let mime = &caps[..semicolon_pos];
                                if let Some(comma_pos) = caps.find(',') {
                                    let encoded = &caps[comma_pos + 1..];
                                    json!({
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": encoded
                                    })
                                } else {
                                    return Err(ConversionError::InvalidFormat(
                                        "Invalid data URL format".to_string(),
                                    ));
                                }
                            } else {
                                return Err(ConversionError::InvalidFormat(
                                    "Invalid data URL format".to_string(),
                                ));
                            }
                        } else {
                            return Err(ConversionError::InvalidFormat(
                                "Invalid data URL format".to_string(),
                            ));
                        }
                    } else {
                        json!({
                            "type": "url",
                            "url": url
                        })
                    }
                } else {
                    return Err(ConversionError::InvalidFormat(
                        "Image content missing URL or data".to_string(),
                    ));
                };

                Ok(json!({
                    "type": "image",
                    "source": source
                }))
            } else {
                Err(ConversionError::InvalidFormat(
                    "Image content missing media".to_string(),
                ))
            }
        }
        ContentType::ToolCall => {
            if let Some(tool_call) = &content.tool_call {
                Ok(json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments
                }))
            } else {
                Err(ConversionError::InvalidFormat(
                    "Tool call content missing tool_call".to_string(),
                ))
            }
        }
        ContentType::ToolResult => {
            if let Some(tool_result) = &content.tool_result {
                Ok(json!({
                    "type": "tool_result",
                    "tool_use_id": tool_result.tool_call_id,
                    "content": tool_result.result
                }))
            } else {
                Err(ConversionError::InvalidFormat(
                    "Tool result content missing tool_result".to_string(),
                ))
            }
        }
        _ => Err(ConversionError::InvalidFormat(format!(
            "Unsupported content type for Anthropic: {:?}",
            content.content_type
        ))),
    }
}

fn convert_universal_tool_to_anthropic(tool: &UniversalTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters
    })
}

fn convert_tool_choice_to_anthropic(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::String(s) => match s.as_str() {
            "auto" => json!({"type": "auto"}),
            "required" | "any" => json!({"type": "any"}),
            "none" => json!({"type": "auto"}), // Anthropic doesn't have "none"
            _ => json!({"type": "auto"}),
        },
        ToolChoice::Named { name } => json!({
            "type": "tool",
            "name": name
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_anthropic_conversion() {
        let anthropic_body = json!({
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "system": "You are a helpful assistant",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        });

        let universal = anthropic_to_universal(anthropic_body).unwrap();

        assert_eq!(universal.provider, ProviderType::Anthropic);
        assert_eq!(universal.model, "claude-3-opus-20240229");
        assert_eq!(universal.max_tokens, Some(1024));
        assert!(matches!(
            universal.system,
            Some(UniversalSystemPrompt::String(_))
        ));
        assert_eq!(universal.messages.len(), 1);
        assert_eq!(universal.messages[0].role, UniversalRole::User);
    }

    #[test]
    fn test_anthropic_image_content() {
        let anthropic_body = json!({
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "iVBORw0KGgoAAAANS..."
                        }
                    }
                ]
            }]
        });

        let universal = anthropic_to_universal(anthropic_body).unwrap();

        assert_eq!(universal.messages[0].content.len(), 2);
        assert_eq!(
            universal.messages[0].content[0].content_type,
            ContentType::Text
        );
        assert_eq!(
            universal.messages[0].content[1].content_type,
            ContentType::Image
        );

        let media = universal.messages[0].content[1].media.as_ref().unwrap();
        assert_eq!(media.mime_type, Some("image/jpeg".to_string()));
        assert!(media.data.is_some());
    }
}
