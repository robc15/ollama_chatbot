# AI Chatbot with Multiple Model Support

A Streamlit-based AI chatbot application that provides access to multiple AI models including OpenAI GPT models and local Ollama models. The application supports multimodal interactions with text, PDF, and image file uploads.

## Features

- **Multiple AI Models**: Support for GPT-4, GPT-4o, Claude models, and local models via Ollama (Llama 3, Gemma, DeepSeek)
- **Multimodal Support**: Text, PDF, and image file analysis for compatible models
- **Cost Transparency**: Real-time pricing information for each model
- **Local Processing**: Privacy-focused local model execution with Ollama
- **File Upload**: Support for various file types including PDF, text, and images
- **Interactive UI**: Clean Streamlit interface with model selection and file management

## Supported Models

### Cloud Models
- **GPT-4.1**: Advanced reasoning and complex tasks
- **GPT-4o**: Multimodal capabilities (text, vision, audio)
- **GPT-4o Mini**: Cost-effective version of GPT-4o
- **Claude 3.5 Sonnet**: Advanced writing, analysis, and coding capabilities
- **Claude 3.5 Haiku**: Fast and efficient Claude model
- **Claude Sonnet 4**: Anthropic's most advanced model with superior reasoning

### Local Models (via Ollama)
- **Llama 3**: Meta's open-source model
- **Gemma**: Google's lightweight model
- **DeepSeek**: Coding and reasoning focused model

## Additional Components

- **fine_tune.py**: GPT-2 fine-tuning script using Hugging Face Transformers for custom model training

## Configuration

### Required Environment Variables
- **OPENAI_API_KEY**: Required for OpenAI GPT models (GPT-4.1, GPT-4o, GPT-4o Mini)
- **CLAUDE_API_KEY**: Required for Claude models (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude Sonnet 4)

### Local Models Setup
For local models, install Ollama and pull the desired models:
```bash
ollama pull llama3
ollama pull gemma
ollama pull deepseek-r1
```

## Usage

1. Set your API keys:
   - OpenAI: `export OPENAI_API_KEY="your-openai-api-key"`
   - Claude: `export CLAUDE_API_KEY="your-claude-api-key"`
2. Run the Streamlit application: `streamlit run chatgpt.py`
3. Select your preferred AI model from the sidebar
4. Enter your question or upload a file for analysis
5. Click "Ask" to get responses from the selected model

The application runs on port 8502 and provides detailed pricing information for each model to help users make informed choices about cost and capabilities.
