use crate::error::{ConversionError, Result};
use crate::types::*;
use crate::utils::generate_id;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Convert Google format to Universal format
pub fn google_to_universal(body: Value) -> Result<UniversalBody> {
    if !body.is_object() {
        return Err(ConversionError::InvalidFormat(
            "Request body must be an object".to_string(),
        ));
    }

    let obj = body.as_object().unwrap();

    // Extract contents (Google's version of messages)
    let contents = obj
        .get("contents")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ConversionError::MissingField("contents".to_string()))?;

    // Extract system instruction if present
    let system = obj
        .get("systemInstruction")
        .or_else(|| obj.get("system_instruction"))
        .and_then(|si| {
            si.get("parts")
                .and_then(|p| p.as_array())
                .and_then(|parts| {
                    let text: String = parts
                        .iter()
                        .filter_map(|part| part.get("text")?.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !text.is_empty() {
                        Some(UniversalSystemPrompt::String(text))
                    } else {
                        None
                    }
                })
        });

    // Convert contents to messages
    let universal_messages: Result<Vec<UniversalMessage>> = contents
        .iter()
        .enumerate()
        .map(|(index, content)| parse_google_content(content, index))
        .collect();

    // Extract tools
    let tools = obj
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools_array| {
            tools_array
                .iter()
                .filter_map(|tool| parse_google_tool(tool).ok())
                .flatten()
                .collect()
        });

    // Model is typically not in the request body for Google, but we check anyway
    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("gemini-pro")
        .to_string();

    // Extract generation config
    let gen_config = obj
        .get("generationConfig")
        .or_else(|| obj.get("generation_config"));

    // Extract tool config for tool_choice
    let tool_choice = obj
        .get("toolConfig")
        .or_else(|| obj.get("tool_config"))
        .and_then(|tc| {
            tc.get("functionCallingConfig")
                .or_else(|| tc.get("function_calling_config"))
        })
        .and_then(|fcc| fcc.get("mode"))
        .and_then(|mode| mode.as_str())
        .map(|mode_str| {
            let lower = mode_str.to_lowercase();
            match lower.as_str() {
                "auto" => ToolChoice::String("auto".to_string()),
                "any" => ToolChoice::String("required".to_string()),
                "none" => ToolChoice::String("none".to_string()),
                _ => ToolChoice::String("auto".to_string()),
            }
        });

    Ok(UniversalBody {
        provider: ProviderType::Google,
        system,
        messages: universal_messages?,
        model,
        temperature: gen_config
            .and_then(|gc| gc.get("temperature"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        max_tokens: gen_config
            .and_then(|gc| {
                gc.get("maxOutputTokens")
                    .or_else(|| gc.get("max_output_tokens"))
            })
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        top_p: gen_config
            .and_then(|gc| gc.get("topP").or_else(|| gc.get("top_p")))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        frequency_penalty: gen_config
            .and_then(|gc| {
                gc.get("frequencyPenalty")
                    .or_else(|| gc.get("frequency_penalty"))
            })
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        presence_penalty: gen_config
            .and_then(|gc| {
                gc.get("presencePenalty")
                    .or_else(|| gc.get("presence_penalty"))
            })
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        seed: None,
        stream: None,
        tools,
        tool_choice,
        provider_params: extract_google_params(obj),
        _original: Some(OriginalContent {
            provider: ProviderType::Google,
            raw: body,
        }),
    })
}

/// Convert Universal format to Google format
pub fn universal_to_google(universal: &UniversalBody) -> Result<Value> {
    let mut result = json!({});
    let obj = result.as_object_mut().unwrap();

    // Separate system messages from regular messages
    let system_messages: Vec<_> = universal
        .messages
        .iter()
        .filter(|msg| msg.role == UniversalRole::System)
        .collect();
    let regular_messages: Vec<_> = universal
        .messages
        .iter()
        .filter(|msg| msg.role != UniversalRole::System)
        .collect();

    // Add system instruction if present
    let mut system_parts = Vec::new();

    // Add system from universal.system field
    if let Some(system) = &universal.system {
        let system_content = match system {
            UniversalSystemPrompt::String(s) => s.clone(),
            UniversalSystemPrompt::Complex { content, .. } => content.clone(),
        };
        system_parts.push(json!({"text": system_content}));
    }

    // Add system messages from messages array
    for system_msg in system_messages {
        for content in &system_msg.content {
            if content.content_type == ContentType::Text {
                if let Some(text) = &content.text {
                    system_parts.push(json!({"text": text}));
                }
            }
            // Note: Google system instructions only support text content
        }
    }

    if !system_parts.is_empty() {
        obj.insert(
            "systemInstruction".to_string(),
            json!({
                "parts": system_parts
            }),
        );
    }

    // Convert messages to contents
    let google_contents: Vec<Value> = regular_messages
        .iter()
        .filter_map(|msg| convert_universal_message_to_google(msg).ok())
        .collect();

    obj.insert("contents".to_string(), json!(google_contents));

    // Add generation config
    let mut gen_config = serde_json::Map::new();
    if let Some(temp) = universal.temperature {
        gen_config.insert("temperature".to_string(), json!(temp));
    }
    if let Some(max_tokens) = universal.max_tokens {
        gen_config.insert("maxOutputTokens".to_string(), json!(max_tokens));
    }
    if let Some(top_p) = universal.top_p {
        gen_config.insert("topP".to_string(), json!(top_p));
    }
    if let Some(freq_penalty) = universal.frequency_penalty {
        gen_config.insert("frequencyPenalty".to_string(), json!(freq_penalty));
    }
    if let Some(pres_penalty) = universal.presence_penalty {
        gen_config.insert("presencePenalty".to_string(), json!(pres_penalty));
    }

    if !gen_config.is_empty() {
        obj.insert("generationConfig".to_string(), json!(gen_config));
    }

    // Add tools if present
    if let Some(tools) = &universal.tools {
        let google_tools: Vec<Value> = vec![json!({
            "functionDeclarations": tools
                .iter()
                .map(convert_universal_tool_to_google)
                .collect::<Vec<_>>()
        })];
        obj.insert("tools".to_string(), json!(google_tools));

        // Add tool config
        if let Some(tool_choice) = &universal.tool_choice {
            let mode = match tool_choice {
                ToolChoice::String(s) => match s.as_str() {
                    "auto" => "AUTO",
                    "required" | "any" => "ANY",
                    "none" => "NONE",
                    _ => "AUTO",
                },
                ToolChoice::Named { .. } => "ANY", // Google doesn't support named tool choice
            };

            obj.insert(
                "toolConfig".to_string(),
                json!({
                    "functionCallingConfig": {
                        "mode": mode
                    }
                }),
            );
        }
    }

    // Add provider-specific params
    if let Some(params) = &universal.provider_params {
        for (key, value) in params {
            obj.insert(key.clone(), value.clone());
        }
    }

    Ok(result)
}

fn parse_google_content(content: &Value, index: usize) -> Result<UniversalMessage> {
    let role = content
        .get("role")
        .and_then(|r| r.as_str())
        .ok_or_else(|| ConversionError::MissingField("role".to_string()))?;

    let universal_role = match role {
        "user" => UniversalRole::User,
        "model" => UniversalRole::Assistant,
        "function" => UniversalRole::Tool,
        _ => UniversalRole::User,
    };

    let parts = content
        .get("parts")
        .and_then(|p| p.as_array())
        .ok_or_else(|| ConversionError::MissingField("parts".to_string()))?;

    let universal_content: Vec<UniversalContent> = parts
        .iter()
        .filter_map(|part| parse_google_part(part).ok())
        .collect();

    Ok(UniversalMessage {
        id: generate_id(),
        role: universal_role,
        content: universal_content,
        metadata: MessageMetadata {
            provider: ProviderType::Google,
            original_role: Some(role.to_string()),
            original_index: Some(index),
            cache_control: None,
            name: None,
            tool_call_id: None,
            parts_metadata: Some(parts.to_vec()),
            extra: HashMap::new(),
        },
        tool_calls: None,
    })
}

fn parse_google_part(part: &Value) -> Result<UniversalContent> {
    // Google parts can have different fields - check in order of priority

    // Text content
    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
        return Ok(UniversalContent {
            content_type: ContentType::Text,
            text: Some(text.to_string()),
            media: None,
            tool_call: None,
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::Google,
                raw: part.clone(),
            }),
        });
    }

    // Inline data (images, audio, video, documents)
    if let Some(inline_data) = part.get("inlineData").or_else(|| part.get("inline_data")) {
        let mime_type = inline_data
            .get("mimeType")
            .or_else(|| inline_data.get("mime_type"))
            .and_then(|m| m.as_str())
            .map(|s| s.to_string());
        let data = inline_data
            .get("data")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());

        // Determine content type based on MIME type
        let content_type = if let Some(ref mime) = mime_type {
            if mime.starts_with("image/") {
                ContentType::Image
            } else if mime.starts_with("audio/") {
                ContentType::Audio
            } else if mime.starts_with("video/") {
                ContentType::Video
            } else {
                ContentType::Document
            }
        } else {
            ContentType::Document
        };

        return Ok(UniversalContent {
            content_type,
            text: None,
            media: Some(UniversalMediaContent {
                url: None,
                detail: None,
                data,
                mime_type,
                file_uri: None,
                file_name: part
                    .get("fileName")
                    .and_then(|f| f.as_str())
                    .map(|s| s.to_string()),
                size: None,
                duration: None,
                metadata: None,
            }),
            tool_call: None,
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::Google,
                raw: part.clone(),
            }),
        });
    }

    // File data (Google Cloud Storage files)
    if let Some(file_data) = part.get("fileData").or_else(|| part.get("file_data")) {
        let mime_type = file_data
            .get("mimeType")
            .or_else(|| file_data.get("mime_type"))
            .and_then(|m| m.as_str())
            .map(|s| s.to_string());
        let file_uri = file_data
            .get("fileUri")
            .or_else(|| file_data.get("file_uri"))
            .and_then(|u| u.as_str())
            .map(|s| s.to_string());
        let file_name = file_data
            .get("fileName")
            .or_else(|| file_data.get("file_name"))
            .and_then(|f| f.as_str())
            .map(|s| s.to_string())
            .or_else(|| Some("document".to_string()));

        return Ok(UniversalContent {
            content_type: ContentType::Document,
            text: None,
            media: Some(UniversalMediaContent {
                url: None,
                detail: None,
                data: None,
                mime_type,
                file_uri,
                file_name,
                size: None,
                duration: None,
                metadata: None,
            }),
            tool_call: None,
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::Google,
                raw: part.clone(),
            }),
        });
    }

    // Function call
    if let Some(function_call) = part
        .get("functionCall")
        .or_else(|| part.get("function_call"))
    {
        let name = function_call
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| {
                ConversionError::InvalidFormat("Function call missing name".to_string())
            })?;

        let args = function_call.get("args").cloned().unwrap_or(json!({}));
        let arguments: HashMap<String, Value> = if let Some(obj) = args.as_object() {
            obj.clone().into_iter().collect()
        } else {
            HashMap::new()
        };

        let mut metadata = HashMap::new();
        metadata.insert("args".to_string(), args);

        return Ok(UniversalContent {
            content_type: ContentType::ToolCall,
            text: None,
            media: None,
            tool_call: Some(UniversalToolCall {
                id: format!("call_{}", generate_id()),
                name: name.to_string(),
                arguments,
                metadata: Some(metadata),
            }),
            tool_result: None,
            _original: Some(OriginalContent {
                provider: ProviderType::Google,
                raw: part.clone(),
            }),
        });
    }

    // Function response
    if let Some(function_response) = part
        .get("functionResponse")
        .or_else(|| part.get("function_response"))
    {
        let name = function_response
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("");
        let response = function_response
            .get("response")
            .cloned()
            .unwrap_or(json!(null));

        let mut metadata = HashMap::new();
        metadata.insert("response".to_string(), response.clone());

        return Ok(UniversalContent {
            content_type: ContentType::ToolResult,
            text: None,
            media: None,
            tool_call: None,
            tool_result: Some(UniversalToolResult {
                tool_call_id: format!("call_{}", name),
                name: name.to_string(),
                result: response,
                error: None,
                metadata: Some(metadata),
            }),
            _original: Some(OriginalContent {
                provider: ProviderType::Google,
                raw: part.clone(),
            }),
        });
    }

    // Fallback for unknown parts
    Err(ConversionError::InvalidFormat(
        "Unknown Google part type".to_string(),
    ))
}

fn parse_google_tool(tool: &Value) -> Result<Vec<UniversalTool>> {
    // Google tools have functionDeclarations
    if let Some(function_declarations) = tool
        .get("functionDeclarations")
        .or_else(|| tool.get("function_declarations"))
        .and_then(|fd| fd.as_array())
    {
        let tools: Vec<UniversalTool> = function_declarations
            .iter()
            .filter_map(|func| {
                let name = func.get("name")?.as_str()?;
                let description = func
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");
                let parameters = func.get("parameters").cloned().unwrap_or(json!({}));

                let mut metadata = HashMap::new();
                metadata.insert("function_declarations".to_string(), json!([func]));

                Some(UniversalTool {
                    name: name.to_string(),
                    description: description.to_string(),
                    parameters,
                    metadata: Some(metadata),
                    _original: Some(OriginalContent {
                        provider: ProviderType::Google,
                        raw: tool.clone(),
                    }),
                })
            })
            .collect();

        Ok(tools)
    } else {
        Ok(vec![])
    }
}

fn extract_google_params(obj: &serde_json::Map<String, Value>) -> Option<HashMap<String, Value>> {
    let mut params = HashMap::new();

    if let Some(safety_settings) = obj
        .get("safetySettings")
        .or_else(|| obj.get("safety_settings"))
    {
        params.insert("safety_settings".to_string(), safety_settings.clone());
    }
    if let Some(generation_config) = obj
        .get("generationConfig")
        .or_else(|| obj.get("generation_config"))
    {
        params.insert("generation_config".to_string(), generation_config.clone());
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

fn convert_universal_message_to_google(msg: &UniversalMessage) -> Result<Value> {
    let role = match msg.role {
        UniversalRole::User => "user",
        UniversalRole::Assistant => "model",
        UniversalRole::Tool => "function",
        _ => "user",
    };

    let parts: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|c| convert_universal_content_to_google(c).ok())
        .collect();

    Ok(json!({
        "role": role,
        "parts": parts
    }))
}

fn convert_universal_content_to_google(content: &UniversalContent) -> Result<Value> {
    match content.content_type {
        ContentType::Text => Ok(json!({
            "text": content.text.as_ref().unwrap_or(&String::new())
        })),
        ContentType::Image | ContentType::Audio | ContentType::Video | ContentType::Document => {
            if let Some(media) = &content.media {
                // Check for file URI (Google Cloud Storage)
                if let Some(file_uri) = &media.file_uri {
                    return Ok(json!({
                        "fileData": {
                            "mimeType": media.mime_type.as_ref().unwrap_or(&"application/octet-stream".to_string()),
                            "fileUri": file_uri
                        }
                    }));
                }

                // Otherwise use inline data
                if let (Some(data), Some(mime_type)) = (&media.data, &media.mime_type) {
                    Ok(json!({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data
                        }
                    }))
                } else if let Some(url) = &media.url {
                    // Try to extract from data URL
                    if url.starts_with("data:") {
                        if let Some(caps) = url.strip_prefix("data:") {
                            if let Some(semicolon_pos) = caps.find(';') {
                                let mime = &caps[..semicolon_pos];
                                if let Some(comma_pos) = caps.find(',') {
                                    let encoded = &caps[comma_pos + 1..];
                                    return Ok(json!({
                                        "inlineData": {
                                            "mimeType": mime,
                                            "data": encoded
                                        }
                                    }));
                                }
                            }
                        }
                    }
                    Err(ConversionError::InvalidFormat(
                        "Google requires base64 data or file URI, not URLs".to_string(),
                    ))
                } else {
                    Err(ConversionError::InvalidFormat(
                        "Media content missing data or file URI".to_string(),
                    ))
                }
            } else {
                Err(ConversionError::InvalidFormat(
                    "Media content missing media field".to_string(),
                ))
            }
        }
        ContentType::ToolCall => {
            if let Some(tool_call) = &content.tool_call {
                Ok(json!({
                    "functionCall": {
                        "name": tool_call.name,
                        "args": tool_call.arguments
                    }
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
                    "functionResponse": {
                        "name": tool_result.name,
                        "response": tool_result.result
                    }
                }))
            } else {
                Err(ConversionError::InvalidFormat(
                    "Tool result content missing tool_result".to_string(),
                ))
            }
        }
    }
}

fn convert_universal_tool_to_google(tool: &UniversalTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_google_conversion() {
        let google_body = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello!"}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        });

        let universal = google_to_universal(google_body).unwrap();

        assert_eq!(universal.provider, ProviderType::Google);
        assert_eq!(universal.temperature, Some(0.7));
        assert_eq!(universal.max_tokens, Some(1000));
        assert_eq!(universal.messages.len(), 1);
        assert_eq!(universal.messages[0].role, UniversalRole::User);
    }

    #[test]
    fn test_google_multimodal() {
        let google_body = json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "What's in this image?"},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": "base64encodeddata"
                        }
                    }
                ]
            }]
        });

        let universal = google_to_universal(google_body).unwrap();

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
        assert_eq!(media.data, Some("base64encodeddata".to_string()));
    }

    #[test]
    fn test_google_audio_video() {
        let google_body = json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/mp3",
                            "data": "audiodata"
                        }
                    },
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "data": "videodata"
                        }
                    }
                ]
            }]
        });

        let universal = google_to_universal(google_body).unwrap();

        assert_eq!(universal.messages[0].content.len(), 2);
        assert_eq!(
            universal.messages[0].content[0].content_type,
            ContentType::Audio
        );
        assert_eq!(
            universal.messages[0].content[1].content_type,
            ContentType::Video
        );
    }

    #[test]
    fn test_google_tools() {
        let google_body = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Get the weather"}]
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

        let universal = google_to_universal(google_body).unwrap();

        assert!(universal.tools.is_some());
        let tools = universal.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
    }

    #[test]
    fn test_google_system_instruction() {
        let google_body = json!({
            "systemInstruction": {
                "parts": [{"text": "You are helpful"}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello"}]
            }]
        });

        let universal = google_to_universal(google_body).unwrap();

        assert!(matches!(
            universal.system,
            Some(UniversalSystemPrompt::String(_))
        ));
    }
}
