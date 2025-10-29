use crate::error::{ConversionError, Result};
use crate::types::*;
use crate::utils::generate_id;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Convert OpenAI format to Universal format
pub fn openai_to_universal(body: Value) -> Result<UniversalBody> {
    // Validate input
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

    // Extract system prompt and filter messages
    let system_prompt = extract_system_from_openai_messages(messages);
    let non_system_messages: Vec<_> = messages
        .iter()
        .filter(|msg| {
            msg.get("role")
                .and_then(|r| r.as_str())
                .map(|r| r != "system")
                .unwrap_or(true)
        })
        .collect();

    // Convert messages to universal format
    let universal_messages: Result<Vec<UniversalMessage>> = non_system_messages
        .iter()
        .enumerate()
        .map(|(index, msg)| parse_openai_message(msg, index))
        .collect();

    // Extract tools
    let tools = obj
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools_array| {
            tools_array
                .iter()
                .filter_map(|tool| parse_openai_tool(tool).ok())
                .collect()
        });

    // Extract model
    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(UniversalBody {
        provider: ProviderType::OpenAI,
        system: system_prompt,
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
        frequency_penalty: obj
            .get("frequency_penalty")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        presence_penalty: obj
            .get("presence_penalty")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        seed: obj.get("seed").and_then(|v| v.as_u64()).map(|v| v as u32),
        stream: obj.get("stream").and_then(|v| v.as_bool()),
        tools,
        tool_choice: obj.get("tool_choice").map(parse_tool_choice),
        provider_params: extract_openai_params(obj),
        _original: Some(OriginalContent {
            provider: ProviderType::OpenAI,
            raw: body,
        }),
    })
}

/// Convert Universal format to OpenAI format
pub fn universal_to_openai(universal: &UniversalBody) -> Result<Value> {
    let mut messages = Vec::new();

    // Add system message if present
    if let Some(system) = &universal.system {
        let system_content = match system {
            UniversalSystemPrompt::String(s) => s.clone(),
            UniversalSystemPrompt::Complex { content, .. } => content.clone(),
        };
        messages.push(json!({
            "role": "system",
            "content": system_content
        }));
    }

    // Convert universal messages to OpenAI format
    for msg in &universal.messages {
        let openai_msg = convert_universal_message_to_openai(msg)?;
        messages.push(openai_msg);
    }

    let mut result = json!({
        "model": universal.model,
        "messages": messages
    });

    let obj = result.as_object_mut().unwrap();

    // Add optional parameters
    if let Some(temp) = universal.temperature {
        obj.insert("temperature".to_string(), json!(temp));
    }
    if let Some(max_tokens) = universal.max_tokens {
        obj.insert("max_tokens".to_string(), json!(max_tokens));
    }
    if let Some(top_p) = universal.top_p {
        obj.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(freq_penalty) = universal.frequency_penalty {
        obj.insert("frequency_penalty".to_string(), json!(freq_penalty));
    }
    if let Some(pres_penalty) = universal.presence_penalty {
        obj.insert("presence_penalty".to_string(), json!(pres_penalty));
    }
    if let Some(seed) = universal.seed {
        obj.insert("seed".to_string(), json!(seed));
    }
    if let Some(stream) = universal.stream {
        obj.insert("stream".to_string(), json!(stream));
    }

    // Add tools if present
    if let Some(tools) = &universal.tools {
        let openai_tools: Vec<Value> = tools.iter().map(convert_universal_tool_to_openai).collect();
        obj.insert("tools".to_string(), json!(openai_tools));
    }

    // Add tool_choice if present
    if let Some(tool_choice) = &universal.tool_choice {
        obj.insert(
            "tool_choice".to_string(),
            convert_tool_choice_to_openai(tool_choice),
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

fn extract_system_from_openai_messages(messages: &[Value]) -> Option<UniversalSystemPrompt> {
    messages
        .iter()
        .find(|msg| {
            msg.get("role")
                .and_then(|r| r.as_str())
                .map(|r| r == "system")
                .unwrap_or(false)
        })
        .and_then(|msg| {
            msg.get("content").and_then(|content| {
                if let Some(s) = content.as_str() {
                    Some(UniversalSystemPrompt::String(s.to_string()))
                } else if let Some(arr) = content.as_array() {
                    // Join text parts
                    let text: String = arr
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
                    if !text.is_empty() {
                        Some(UniversalSystemPrompt::String(text))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        })
}

fn parse_openai_message(msg: &Value, index: usize) -> Result<UniversalMessage> {
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .ok_or_else(|| ConversionError::MissingField("role".to_string()))?;

    let universal_role = match role {
        "user" => UniversalRole::User,
        "assistant" => UniversalRole::Assistant,
        "tool" => UniversalRole::Tool,
        "system" => UniversalRole::System,
        "developer" => UniversalRole::Developer,
        _ => UniversalRole::User,
    };

    let content = parse_openai_content(msg.get("content"))?;

    let metadata = MessageMetadata {
        provider: ProviderType::OpenAI,
        original_role: None,
        original_index: Some(index),
        cache_control: None,
        name: msg
            .get("name")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string()),
        tool_call_id: msg
            .get("tool_call_id")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string()),
        parts_metadata: None,
        extra: HashMap::new(),
    };

    // Parse tool calls if present
    let tool_calls = msg
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|tcs| {
            tcs.iter()
                .filter_map(|tc| parse_openai_tool_call(tc).ok())
                .collect()
        });

    Ok(UniversalMessage {
        id: generate_id(),
        role: universal_role,
        content,
        metadata,
        tool_calls,
    })
}

fn parse_openai_content(content: Option<&Value>) -> Result<Vec<UniversalContent>> {
    match content {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::String(s)) => Ok(vec![UniversalContent {
            content_type: ContentType::Text,
            text: Some(s.clone()),
            media: None,
            tool_call: None,
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::OpenAI,
                raw: json!(s),
            }),
        }]),
        Some(Value::Array(arr)) => {
            let contents: Vec<UniversalContent> = arr
                .iter()
                .filter_map(|part| parse_openai_content_part(part).ok())
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
                provider: ProviderType::OpenAI,
                raw: other.clone(),
            }),
        }]),
    }
}

fn parse_openai_content_part(part: &Value) -> Result<UniversalContent> {
    let part_type = part
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Content part missing type".to_string()))?;

    match part_type {
        "text" => {
            let text = part.get("text").and_then(|t| t.as_str()).ok_or_else(|| {
                ConversionError::InvalidFormat("Text part missing text".to_string())
            })?;
            Ok(UniversalContent {
                content_type: ContentType::Text,
                text: Some(text.to_string()),
                media: None,
                tool_call: None,
                tool_result: None,
                _original: Some(OriginalContent {
                    provider: ProviderType::OpenAI,
                    raw: part.clone(),
                }),
            })
        }
        "image_url" => {
            let image_url = part.get("image_url").ok_or_else(|| {
                ConversionError::InvalidFormat("Image part missing image_url".to_string())
            })?;
            let url = image_url
                .get("url")
                .and_then(|u| u.as_str())
                .ok_or_else(|| {
                    ConversionError::InvalidFormat("Image URL missing url".to_string())
                })?;

            // Extract MIME type and data from data URLs
            let (mime_type, data) = if url.starts_with("data:") {
                if let Some(caps) = url.strip_prefix("data:") {
                    if let Some(semicolon_pos) = caps.find(';') {
                        let mime = &caps[..semicolon_pos];
                        if let Some(comma_pos) = caps.find(',') {
                            let encoded = &caps[comma_pos + 1..];
                            (Some(mime.to_string()), Some(encoded.to_string()))
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            Ok(UniversalContent {
                content_type: ContentType::Image,
                text: None,
                media: Some(UniversalMediaContent {
                    url: Some(url.to_string()),
                    detail: image_url
                        .get("detail")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    data,
                    mime_type,
                    file_uri: None,
                    file_name: None,
                    size: None,
                    duration: None,
                    metadata: None,
                }),
                tool_call: None,
                tool_result: None,
                _original: Some(OriginalContent {
                    provider: ProviderType::OpenAI,
                    raw: part.clone(),
                }),
            })
        }
        _ => Err(ConversionError::InvalidFormat(format!(
            "Unknown content part type: {}",
            part_type
        ))),
    }
}

fn parse_openai_tool_call(tc: &Value) -> Result<UniversalToolCall> {
    let id = tc
        .get("id")
        .and_then(|i| i.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Tool call missing id".to_string()))?;

    let function = tc
        .get("function")
        .ok_or_else(|| ConversionError::InvalidFormat("Tool call missing function".to_string()))?;

    let name = function
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Function missing name".to_string()))?;

    let arguments_str = function
        .get("arguments")
        .and_then(|a| a.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Function missing arguments".to_string()))?;

    let arguments: HashMap<String, Value> = serde_json::from_str(arguments_str)
        .map_err(|e| ConversionError::InvalidToolArguments(e.to_string()))?;

    let mut metadata = HashMap::new();
    if let Some(tc_type) = tc.get("type") {
        metadata.insert("type".to_string(), tc_type.clone());
    }

    Ok(UniversalToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
        metadata: Some(metadata),
    })
}

fn parse_openai_tool(tool: &Value) -> Result<UniversalTool> {
    let tool_type = tool.get("type").and_then(|t| t.as_str());

    let function = tool
        .get("function")
        .ok_or_else(|| ConversionError::InvalidFormat("Tool missing function".to_string()))?;

    let name = function
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| ConversionError::InvalidFormat("Function missing name".to_string()))?;

    let description = function
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("");

    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let mut metadata = HashMap::new();
    if let Some(tt) = tool_type {
        metadata.insert("type".to_string(), json!(tt));
    }

    Ok(UniversalTool {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
        metadata: Some(metadata),
        _original: Some(OriginalContent {
            provider: ProviderType::OpenAI,
            raw: tool.clone(),
        }),
    })
}

fn parse_tool_choice(tc: &Value) -> ToolChoice {
    if let Some(s) = tc.as_str() {
        ToolChoice::String(s.to_string())
    } else if let Some(obj) = tc.as_object() {
        if let Some(name) = obj
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        {
            ToolChoice::Named {
                name: name.to_string(),
            }
        } else {
            ToolChoice::String("auto".to_string())
        }
    } else {
        ToolChoice::String("auto".to_string())
    }
}

fn extract_openai_params(obj: &serde_json::Map<String, Value>) -> Option<HashMap<String, Value>> {
    let mut params = HashMap::new();

    // OpenAI-specific parameters
    if let Some(response_format) = obj.get("response_format") {
        params.insert("response_format".to_string(), response_format.clone());
    }
    if let Some(logprobs) = obj.get("logprobs") {
        params.insert("logprobs".to_string(), logprobs.clone());
    }
    if let Some(top_logprobs) = obj.get("top_logprobs") {
        params.insert("top_logprobs".to_string(), top_logprobs.clone());
    }
    if let Some(n) = obj.get("n") {
        params.insert("n".to_string(), n.clone());
    }
    if let Some(stop) = obj.get("stop") {
        params.insert("stop".to_string(), stop.clone());
    }
    if let Some(user) = obj.get("user") {
        params.insert("user".to_string(), user.clone());
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

fn convert_universal_message_to_openai(msg: &UniversalMessage) -> Result<Value> {
    let role = match msg.role {
        UniversalRole::User => "user",
        UniversalRole::Assistant => "assistant",
        UniversalRole::Tool => "tool",
        UniversalRole::System => "system",
        UniversalRole::Developer => "developer",
    };

    let mut message = json!({
        "role": role
    });

    let obj = message.as_object_mut().unwrap();

    // Handle tool responses specially
    if msg.role == UniversalRole::Tool {
        if let Some(tool_call_id) = &msg.metadata.tool_call_id {
            obj.insert("tool_call_id".to_string(), json!(tool_call_id));
        }
        if let Some(name) = &msg.metadata.name {
            obj.insert("name".to_string(), json!(name));
        }
        // For tool role, content should be the result
        if !msg.content.is_empty() {
            if let Some(tool_result) = &msg.content[0].tool_result {
                obj.insert(
                    "content".to_string(),
                    json!(serde_json::to_string(&tool_result.result).unwrap_or_default()),
                );
            } else if let Some(text) = &msg.content[0].text {
                obj.insert("content".to_string(), json!(text));
            }
        }
    } else {
        // Convert content
        if msg.content.len() == 1 && msg.content[0].content_type == ContentType::Text {
            // Simple text content
            if let Some(text) = &msg.content[0].text {
                obj.insert("content".to_string(), json!(text));
            }
        } else if !msg.content.is_empty() {
            // Complex content (multimodal)
            let content_parts: Vec<Value> = msg
                .content
                .iter()
                .filter_map(|c| convert_universal_content_to_openai(c).ok())
                .collect();
            obj.insert("content".to_string(), json!(content_parts));
        } else if msg.tool_calls.is_some() {
            // Tool calls with null content
            obj.insert("content".to_string(), Value::Null);
        }

        // Add tool calls if present
        if let Some(tool_calls) = &msg.tool_calls {
            let openai_tool_calls: Vec<Value> = tool_calls
                .iter()
                .map(convert_universal_tool_call_to_openai)
                .collect();
            obj.insert("tool_calls".to_string(), json!(openai_tool_calls));
        }
    }

    // Add name if present and not tool role
    if msg.role != UniversalRole::Tool {
        if let Some(name) = &msg.metadata.name {
            obj.insert("name".to_string(), json!(name));
        }
    }

    Ok(message)
}

fn convert_universal_content_to_openai(content: &UniversalContent) -> Result<Value> {
    match content.content_type {
        ContentType::Text => Ok(json!({
            "type": "text",
            "text": content.text.as_ref().unwrap_or(&String::new())
        })),
        ContentType::Image => {
            if let Some(media) = &content.media {
                let mut image_url = serde_json::Map::new();

                // Reconstruct data URL if we have data and mime type
                let url = if let (Some(data), Some(mime_type)) = (&media.data, &media.mime_type) {
                    format!("data:{};base64,{}", mime_type, data)
                } else if let Some(url) = &media.url {
                    url.clone()
                } else {
                    return Err(ConversionError::InvalidFormat(
                        "Image content missing URL or data".to_string(),
                    ));
                };

                image_url.insert("url".to_string(), json!(url));

                if let Some(detail) = &media.detail {
                    image_url.insert("detail".to_string(), json!(detail));
                }

                Ok(json!({
                    "type": "image_url",
                    "image_url": image_url
                }))
            } else {
                Err(ConversionError::InvalidFormat(
                    "Image content missing media".to_string(),
                ))
            }
        }
        _ => Err(ConversionError::InvalidFormat(format!(
            "Unsupported content type for OpenAI: {:?}",
            content.content_type
        ))),
    }
}

fn convert_universal_tool_call_to_openai(tc: &UniversalToolCall) -> Value {
    json!({
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string())
        }
    })
}

fn convert_universal_tool_to_openai(tool: &UniversalTool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
    })
}

fn convert_tool_choice_to_openai(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::String(s) => json!(s),
        ToolChoice::Named { name } => json!({
            "type": "function",
            "function": {
                "name": name
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_openai_conversion() {
        let openai_body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        });

        let universal = openai_to_universal(openai_body).unwrap();

        assert_eq!(universal.provider, ProviderType::OpenAI);
        assert_eq!(universal.model, "gpt-4");
        assert_eq!(universal.temperature, Some(0.7));
        assert_eq!(universal.max_tokens, Some(1000));
        assert!(matches!(
            universal.system,
            Some(UniversalSystemPrompt::String(_))
        ));
        assert_eq!(universal.messages.len(), 1);
        assert_eq!(universal.messages[0].role, UniversalRole::User);
    }

    #[test]
    fn test_openai_roundtrip() {
        let original = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        });

        let universal = openai_to_universal(original.clone()).unwrap();
        let reconstructed = universal_to_openai(&universal).unwrap();

        assert_eq!(reconstructed.get("model"), original.get("model"));
        assert!(reconstructed.get("messages").is_some());
    }

    #[test]
    fn test_openai_multimodal() {
        let openai_body = json!({
            "model": "gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg",
                            "detail": "high"
                        }
                    }
                ]
            }]
        });

        let universal = openai_to_universal(openai_body).unwrap();

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
        assert_eq!(media.url, Some("https://example.com/image.jpg".to_string()));
        assert_eq!(media.detail, Some("high".to_string()));
    }
}
