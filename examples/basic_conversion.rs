//! Basic example demonstrating conversion between LLM provider formats

use llm_conversion::{from_universal, to_universal, translate_between_providers, ProviderType};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLM Conversion Library Example ===\n");

    // Example 1: OpenAI to Universal
    println!("1. Converting OpenAI format to Universal format:");
    let openai_request = json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    });
    println!("OpenAI Request: {}", serde_json::to_string_pretty(&openai_request)?);

    let universal = to_universal(ProviderType::OpenAI, openai_request.clone())?;
    println!("\nUniversal format provider: {}", universal.provider);
    println!("Universal format model: {}", universal.model);
    println!("Universal format messages: {} messages", universal.messages.len());

    // Example 2: Universal to Anthropic
    println!("\n2. Converting Universal format to Anthropic format:");
    let anthropic_request = from_universal(ProviderType::Anthropic, &universal)?;
    println!("Anthropic Request: {}", serde_json::to_string_pretty(&anthropic_request)?);

    // Example 3: Direct translation
    println!("\n3. Direct translation from OpenAI to Google:");
    let google_request = translate_between_providers(
        ProviderType::OpenAI,
        ProviderType::Google,
        openai_request,
    )?;
    println!("Google Request: {}", serde_json::to_string_pretty(&google_request)?);

    // Example 4: Multimodal content
    println!("\n4. Converting multimodal content (OpenAI to Anthropic):");
    let multimodal_request = json!({
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                        "detail": "high"
                    }
                }
            ]
        }]
    });

    let multimodal_universal = to_universal(ProviderType::OpenAI, multimodal_request)?;
    let anthropic_multimodal = from_universal(ProviderType::Anthropic, &multimodal_universal)?;
    println!("Converted multimodal content to Anthropic format");
    println!("Messages: {}", serde_json::to_string_pretty(&anthropic_multimodal)?);

    // Example 5: Tool calling
    println!("\n5. Converting tool calls (OpenAI to Google):");
    let tool_request = json!({
        "model": "gpt-4",
        "messages": [{
            "role": "user",
            "content": "What's the weather in San Francisco?"
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
    });

    let tool_universal = to_universal(ProviderType::OpenAI, tool_request)?;
    let google_tools = from_universal(ProviderType::Google, &tool_universal)?;
    println!("Google Request with tools: {}", serde_json::to_string_pretty(&google_tools)?);

    println!("\n=== All conversions completed successfully! ===");
    Ok(())
}
