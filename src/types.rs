use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported LLM provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    Google,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::Google => write!(f, "google"),
        }
    }
}

/// Universal role types across providers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UniversalRole {
    System,
    User,
    Assistant,
    Tool,
    Developer,
}

/// Media content representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalMediaContent {
    /// URL-based (OpenAI style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// OpenAI image detail
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,

    /// Base64 data (Anthropic, Google style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none", rename = "mimeType")]
    pub mime_type: Option<String>,

    /// File reference (Google Cloud Storage, etc.)
    #[serde(skip_serializing_if = "Option::is_none", rename = "fileUri")]
    pub file_uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none", rename = "fileName")]
    pub file_name: Option<String>,

    /// Size/duration info
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,

    /// For audio/video
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,

    /// Provider-specific media metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Tool call representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalToolCall {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,

    /// Provider-specific tool call metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Tool result representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub result: serde_json::Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Provider-specific result metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Content block in a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalContent {
    #[serde(rename = "type")]
    pub content_type: ContentType,

    /// Text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Media content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<UniversalMediaContent>,

    /// Tool call content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<UniversalToolCall>,

    /// Tool result content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<UniversalToolResult>,

    /// Preserve original content structure
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _original: Option<OriginalContent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Text,
    Image,
    Audio,
    Video,
    Document,
    ToolCall,
    ToolResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OriginalContent {
    pub provider: ProviderType,
    pub raw: serde_json::Value,
}

/// Message in universal format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalMessage {
    pub id: String,
    pub role: UniversalRole,
    pub content: Vec<UniversalContent>,
    pub metadata: MessageMetadata,

    /// Tool calls if this message contains them
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<UniversalToolCall>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub provider: ProviderType,

    #[serde(skip_serializing_if = "Option::is_none", rename = "originalRole")]
    pub original_role: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none", rename = "originalIndex")]
    pub original_index: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none", rename = "cache_control")]
    pub cache_control: Option<HashMap<String, serde_json::Value>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none", rename = "parts_metadata")]
    pub parts_metadata: Option<Vec<serde_json::Value>>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// System prompt representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UniversalSystemPrompt {
    String(String),
    Complex {
        content: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        parts: Option<Vec<SystemPromptPart>>,

        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<HashMap<String, serde_json::Value>>,

        #[serde(skip_serializing_if = "Option::is_none")]
        _original: Option<OriginalContent>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemPromptPart {
    #[serde(rename = "type")]
    pub part_type: SystemPromptPartType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<UniversalMediaContent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemPromptPartType {
    Text,
    Image,
}

/// Tool definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub _original: Option<OriginalContent>,
}

/// Tool choice options
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "auto", "required", "none"
    Named { name: String },
}

/// Main universal body structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalBody {
    pub provider: ProviderType,

    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<UniversalSystemPrompt>,

    pub messages: Vec<UniversalMessage>,

    /// Generation parameters
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<UniversalTool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Provider-specific parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_params: Option<HashMap<String, serde_json::Value>>,

    /// Original request preservation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _original: Option<OriginalContent>,
}
