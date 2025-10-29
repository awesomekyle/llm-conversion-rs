use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Unsupported provider: {0}")]
    UnsupportedProvider(String),

    #[error("Invalid request format: {0}")]
    InvalidFormat(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid tool call arguments: {0}")]
    InvalidToolArguments(String),

    #[error("Provider detection failed: {0}")]
    DetectionFailed(String),
}

pub type Result<T> = std::result::Result<T, ConversionError>;
