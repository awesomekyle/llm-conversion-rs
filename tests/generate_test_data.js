#!/usr/bin/env node

// Integration test generator - creates test data using upstream llm-bridge
// This validates that our Rust implementation matches the TypeScript behavior

const fs = require('fs');
const path = require('path');

// Try to load llm-bridge from vendor directory
let llmBridge;
try {
    const llmBridgePath = path.join(__dirname, '../vendor/llm-bridge');
    // Check if we need to build it
    const distPath = path.join(llmBridgePath, 'dist');
    if (!fs.existsSync(distPath)) {
        console.log('llm-bridge not built, skipping integration test data generation');
        process.exit(0);
    }
    llmBridge = require(llmBridgePath);
} catch (e) {
    console.log('llm-bridge not available, skipping integration test data generation');
    console.log(e.message);
    process.exit(0);
}

const { toUniversal, fromUniversal, translateBetweenProviders } = llmBridge;

// Test cases
const testCases = {
    openai_basic: {
        provider: 'openai',
        body: {
            model: 'gpt-4',
            messages: [
                { role: 'system', content: 'You are a helpful assistant' },
                { role: 'user', content: 'Hello, how are you?' }
            ],
            temperature: 0.7,
            max_tokens: 1000
        }
    },
    openai_multimodal: {
        provider: 'openai',
        body: {
            model: 'gpt-4-vision-preview',
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'What is in this image?' },
                    {
                        type: 'image_url',
                        image_url: {
                            url: 'data:image/jpeg;base64,abc123def456',
                            detail: 'high'
                        }
                    }
                ]
            }]
        }
    },
    anthropic_basic: {
        provider: 'anthropic',
        body: {
            model: 'claude-3-opus-20240229',
            max_tokens: 1024,
            system: 'You are a helpful assistant',
            messages: [
                { role: 'user', content: 'Hello!' }
            ],
            temperature: 0.7
        }
    },
    anthropic_multimodal: {
        provider: 'anthropic',
        body: {
            model: 'claude-3-opus-20240229',
            max_tokens: 1024,
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'What is in this image?' },
                    {
                        type: 'image',
                        source: {
                            type: 'base64',
                            media_type: 'image/jpeg',
                            data: 'abc123def456'
                        }
                    }
                ]
            }]
        }
    },
    google_basic: {
        provider: 'google',
        body: {
            contents: [{
                role: 'user',
                parts: [{ text: 'Hello!' }]
            }],
            generationConfig: {
                temperature: 0.7,
                maxOutputTokens: 1000
            }
        }
    },
    google_multimodal: {
        provider: 'google',
        body: {
            contents: [{
                role: 'user',
                parts: [
                    { text: 'What is in this image?' },
                    {
                        inlineData: {
                            mimeType: 'image/jpeg',
                            data: 'abc123def456'
                        }
                    }
                ]
            }]
        }
    }
};

// Generate test data
const testData = {};

for (const [testName, testCase] of Object.entries(testCases)) {
    try {
        const universal = toUniversal(testCase.provider, testCase.body);
        
        testData[testName] = {
            provider: testCase.provider,
            input: testCase.body,
            universal: universal,
            // Test roundtrip
            roundtrip: fromUniversal(testCase.provider, universal)
        };

        console.log(`✓ Generated test data for: ${testName}`);
    } catch (error) {
        console.error(`✗ Failed to generate test data for ${testName}:`, error.message);
    }
}

// Save test data
const outputPath = path.join(__dirname, 'integration_test_data.json');
fs.writeFileSync(outputPath, JSON.stringify(testData, null, 2));
console.log(`\nTest data saved to: ${outputPath}`);
console.log(`Generated ${Object.keys(testData).length} test cases`);
