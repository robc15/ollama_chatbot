import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import time
import os
import base64
import streamlit.components.v1 as components
import tempfile

# Force Streamlit to run on port 8502
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

# Image capable models: A list of model IDs that are known to support image input (multimodal).
# This list is used to determine if an uploaded image should be processed into base64 data
# and sent to the model in a multimodal format.
IMAGE_CAPABLE_MODELS = ['gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-sonnet-4-20250514']

# System message for better AI responses
SYSTEM_MESSAGE = (
    "You are a helpful, concise assistant that speaks clearly and answers with expertise. "
    "Provide accurate, well-structured responses that directly address the user's question."
)

# Title of the app
st.title("AI Robot 🤖")

# Description of the app
st.write("Enter a question below and get a response from the model!")

# Initialize session state for user input and voice mode
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'voice_mode' not in st.session_state:
    st.session_state.voice_mode = False
if 'use_local_whisper' not in st.session_state:
    st.session_state.use_local_whisper = True  # Default to local for cost savings
if 'audio_counter' not in st.session_state:
    st.session_state.audio_counter = 0

# Voice/Text mode toggle
st.markdown("### 💬 Input Mode")
voice_mode = st.button("🎤 Voice Mode" if not st.session_state.voice_mode else "📝 Text Mode", use_container_width=True)

if voice_mode:
    st.session_state.voice_mode = not st.session_state.voice_mode
    st.rerun()

# Text input (only show when not in voice mode)
if not st.session_state.voice_mode:
    user_input = st.text_area("Your Question", value=st.session_state.user_input, height=150, key="user_input_area")
    # Update session state when text area changes
    if user_input != st.session_state.user_input:
        st.session_state.user_input = user_input
else:
    user_input = st.session_state.user_input


# Cache the model load process to optimize performance (only run once per session)
@st.cache_resource
def load_model():
    # Simulate model loading time (you can replace this with actual model loading if needed)
    time.sleep(2)  # Simulating the model loading time
    return OpenAI()


@st.cache_resource
def load_anthropic_client():
    # Initialize Anthropic client with API key from environment
    time.sleep(1)  # Simulating the model loading time
    return Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


@st.cache_resource
def load_whisper_model():
    """Load Whisper Base model for local transcription"""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import warnings

    # Suppress urllib3 OpenSSL warning
    warnings.filterwarnings("ignore", message=".*urllib3.*")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-base"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        generate_kwargs={"language": "en", "task": "transcribe"}
    )

    return pipe


# Load the models (this will run only once per session)
client = load_model()
anthropic_client = load_anthropic_client()

# Voice input functionality using OpenAI Whisper (only show in voice mode)
if st.session_state.voice_mode:
    st.markdown("### 🎤 Voice Input (Powered by Whisper)")

    def transcribe_audio_with_whisper_api(audio_bytes, openai_client):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()

                # Open the file and send to Whisper
                with open(tmp_file.name, "rb") as audio_file:
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"  # Can be removed to auto-detect language
                    )

                # Clean up the temporary file
                os.unlink(tmp_file.name)

                return transcript.text

        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None

    def transcribe_audio_with_local_whisper(audio_bytes):
        """Transcribe audio using local Whisper Base model"""
        try:
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()

                # Load the local Whisper model (cached)
                whisper_pipe = load_whisper_model()

                # Transcribe the audio file
                result = whisper_pipe(tmp_file.name)

                # Clean up the temporary file
                os.unlink(tmp_file.name)

                return result["text"]

        except Exception as e:
            st.error(f"Local transcription error: {e}")
            return None

    # Use Streamlit's native audio input widget
    st.markdown("**Click 'Browse files' or drag and drop to upload a pre-recorded audio file, or use your device's recorder if available:**")

    # Generate a unique key for the audio input based on session to reset after processing
    audio_key = f"audio_input_{st.session_state.get('audio_counter', 0)}"

    audio_file = st.audio_input("Record your voice", key=audio_key)

    # Process the audio file if one is provided
    if audio_file is not None:
        # Read the audio bytes from the file object
        audio_bytes = audio_file.getvalue()

        # Determine transcription method
        transcription_method = "Local Whisper Base (FREE)" if st.session_state.use_local_whisper else "OpenAI Whisper API"

        # Transcribe the audio
        with st.spinner(f"🔄 Transcribing audio using {transcription_method}..."):
            if st.session_state.use_local_whisper:
                transcription = transcribe_audio_with_local_whisper(audio_bytes)
            else:
                transcription = transcribe_audio_with_whisper_api(audio_bytes, client)

        if transcription:
            st.success("✅ Transcription complete!")

            # Display the transcription
            st.markdown("**Transcribed text:**")
            st.info(transcription)

            # In voice mode, automatically process the input
            st.info("🚀 Automatically sending to AI...")
            st.session_state.user_input = transcription
            # Increment audio counter to reset the component
            st.session_state.audio_counter += 1
            # Force a rerun to trigger the auto-processing
            st.rerun()
        else:
            st.error("❌ Could not transcribe audio. Please try again.")
            # Increment audio counter to allow retry
            st.session_state.audio_counter += 1


# Model options with descriptions and cost per usage
DEFAULT_MODEL_OPTIONS = [
    # Tier 1: Premium Reasoning & Complex Tasks
    {
        "id": "gpt-4.1",
        "name": "OpenAI GPT-4.1",
        "description": (
            "OpenAI's flagship model. Best for advanced reasoning, creativity, and complex tasks. "
            "Excels at coding, analysis, and nuanced conversation."
        ),
        "cost": "$10.00 / 1M input tokens, $30.00 / 1M output tokens",
        "default": True,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4.0",
        "description": (
            "Anthropic's flagship model with improved reasoning and intelligence. "
            "Superior performance on complex problem-solving, creative tasks, and multimodal understanding."
        ),
        "cost": "$3.00 / 1M input tokens, $15.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    {
        "id": "o1-mini",
        "name": "OpenAI o1-mini",
        "description": (
            "OpenAI's reasoning model optimized for STEM tasks. Excels at coding, math, and logical problem-solving. "
            "Faster and more cost-effective than o1 while maintaining strong reasoning capabilities."
        ),
        "cost": "$3.00 / 1M input tokens, $12.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    },
    # Tier 2: Balanced Performance (Multimodal)
    {
        "id": "gpt-4o",
        "name": "OpenAI GPT-4o",
        "description": (
            "OpenAI's Omni model. Multimodal (text, vision), extremely fast, and cost-effective. "
            "Great for real-time applications and balanced performance."
        ),
        "cost": "$5.00 / 1M input tokens, $15.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    {
        "id": "claude-3-5-sonnet-latest",
        "name": "Claude 3.5 Sonnet",
        "description": (
            "Anthropic's best coding model with excellent reasoning. Industry-leading for software development, "
            "code generation, and technical problem-solving."
        ),
        "cost": "$3.00 / 1M input tokens, $15.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    # Tier 3: Budget/High-Volume
    {
        "id": "gpt-4o-mini",
        "name": "OpenAI GPT-4o Mini",
        "description": (
            "Lighter, faster, affordable version of GPT-4o with vision support. "
            "Ideal for high-volume, low-latency tasks where cost is critical."
        ),
        "cost": "$1.00 / 1M input tokens, $2.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    {
        "id": "claude-3-5-haiku-latest",
        "name": "Claude 3.5 Haiku",
        "description": (
            "Anthropic's fastest model with vision support. Optimized for speed and efficiency. "
            "Great for quick responses, coding assistance, and high-volume use cases."
        ),
        "cost": "$0.80 / 1M input tokens, $4.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    # Tier 4: Local/Privacy-Focused (Free)
    {
        "id": "deepseek-r1",
        "name": "DeepSeek R1",
        "description": (
            "Powerful local model from DeepSeek AI with strong reasoning and coding capabilities. "
            "Competitive with GPT-4 on many tasks. Runs locally via Ollama."
        ),
        "cost": "$0 (runs locally via Ollama)",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "llama3.3",
        "name": "Llama 3.3",
        "description": (
            "Meta's latest Llama model, running locally via Ollama. Excellent general-purpose model "
            "for chat, coding, and analysis. Great for privacy and offline use."
        ),
        "cost": "$0 (runs locally via Ollama)",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "qwen2.5-coder",
        "name": "Qwen 2.5 Coder",
        "description": (
            "Alibaba's coding-focused model with strong multilingual capabilities. Excellent for code generation, "
            "debugging, and technical documentation. Runs locally via Ollama."
        ),
        "cost": "$0 (runs locally via Ollama)",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "mistral",
        "name": "Mistral",
        "description": (
            "Well-balanced open-source model from Mistral AI. Strong at reasoning, coding, and general tasks. "
            "Runs locally via Ollama."
        ),
        "cost": "$0 (runs locally via Ollama)",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    }
]


# Pricing tables for each model (keys must match model IDs in DEFAULT_MODEL_OPTIONS)
PRICING_TABLES = {
    "gpt-4.1": [
        ["Input", "$10.00"],
        ["Output", "$30.00"]
    ],
    "gpt-4o": [
        ["Input", "$5.00"],
        ["Output", "$15.00"]
    ],
    "gpt-4o-mini": [
        ["Input", "$1.00"],
        ["Output", "$2.00"]
    ],
    "o1-mini": [
        ["Input", "$3.00"],
        ["Output", "$12.00"]
    ],
    "claude-sonnet-4-20250514": [
        ["Input", "$3.00"],
        ["Output", "$15.00"]
    ],
    "claude-3-5-sonnet-latest": [
        ["Input", "$3.00"],
        ["Output", "$15.00"]
    ],
    "claude-3-5-haiku-latest": [
        ["Input", "$0.80"],
        ["Output", "$4.00"]
    ],
    "deepseek-r1": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ],
    "llama3.3": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ],
    "qwen2.5-coder": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ],
    "mistral": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ]
}


def show_pricing_table(model_id):
    if model_id in PRICING_TABLES:
        st.sidebar.markdown("**Pricing (per 1 million tokens):**")
        st.sidebar.markdown(
            (
                "<small>**What is a token?** A token is a chunk of text (roughly 4 characters or 0.75 words). "
                "For example, 'ChatGPT is great!' is 5 tokens. Pricing is based on the number of tokens processed.<br>"
                "**Input tokens** are the tokens you send to the model (your prompt/question).<br>"
                "**Output tokens** are the tokens generated by the model in its response.<br>"
                "<br>"
                "<b>Average token usage per prompt:</b><br>"
                "- Short question: 10-30 tokens<br>"
                "- Typical prompt: 50-200 tokens<br>"
                "- Long prompt or document: 500-2,000+ tokens<br>"
                "- Model responses are often similar in length to your prompt, but can be longer for detailed answers."
                "</small>"
            ),
            unsafe_allow_html=True
        )
        st.sidebar.table(
            {
                "Type": [row[0] for row in PRICING_TABLES[model_id]],
                "Price": [row[1] for row in PRICING_TABLES[model_id]],
            }
        )


# Helper: fetch available Claude models from Anthropic API
@st.cache_data(ttl=600)
def fetch_claude_models():
    try:
        models = anthropic_client.models.list()
        model_meta = {m["id"]: m for m in DEFAULT_MODEL_OPTIONS}
        available_claude = []
        found_model_ids = set()

        # First try to match exact model IDs from API
        for m in models.data:
            if m.id in model_meta:
                meta = model_meta[m.id]
                available_claude.append({
                    "id": m.id,
                    "name": meta["name"],
                    "description": meta["description"],
                    "cost": meta["cost"],
                    "default": meta["default"],
                    "supported_file_types": meta["supported_file_types"]
                })
                found_model_ids.add(m.id)

        # Always ensure all Claude models from DEFAULT_MODEL_OPTIONS are included
        for claude_model in DEFAULT_MODEL_OPTIONS:
            if claude_model["id"].startswith("claude-") and claude_model["id"] not in found_model_ids:
                available_claude.append(claude_model)

        # If no models found at all, use fallback
        if not available_claude:
            claude_fallback = [m for m in DEFAULT_MODEL_OPTIONS if m["id"].startswith("claude-")]
            available_claude.extend(claude_fallback)

        return available_claude
    except Exception as e:
        st.warning(f"Could not fetch models from Anthropic: {e}")
        # Fallback to hardcoded Claude models
        claude_fallback = [m for m in DEFAULT_MODEL_OPTIONS if m["id"].startswith("claude-")]
        return claude_fallback


# Helper: fetch available models from OpenAI API
@st.cache_data(ttl=600)
def fetch_openai_models():
    try:
        models = client.models.list()
        model_meta = {m["id"]: m for m in DEFAULT_MODEL_OPTIONS}
        available = []
        for m in models.data:
            if m.id in model_meta:
                meta = model_meta[m.id]
                available.append({
                    "id": m.id,
                    "name": meta["name"],
                    "description": meta["description"],
                    "cost": meta["cost"],
                    "default": meta["default"]
                })
        # Always ensure at least the default model is present
        if not any(m["default"] for m in available):
            available.insert(0, DEFAULT_MODEL_OPTIONS[0])

        # Add Claude models right after OpenAI models
        claude_models = fetch_claude_models()
        for claude_model in claude_models:
            if not any(m["id"] == claude_model["id"] for m in available):
                available.append(claude_model)

        # Always add local Ollama models as options
        ollama_models = ["llama3.3", "deepseek-r1", "qwen2.5-coder", "mistral"]
        for ollama_id in ollama_models:
            if not any(m["id"] == ollama_id for m in available):
                ollama_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == ollama_id), None)
                if ollama_meta:
                    available.append(ollama_meta)
        return available
    except Exception as e:
        st.warning(f"Could not fetch models from OpenAI: {e}")
        # Use the properly ordered DEFAULT_MODEL_OPTIONS as fallback
        fallback = DEFAULT_MODEL_OPTIONS.copy()
        return fallback


# Session state for models (to allow deletion)
if "available_models" not in st.session_state:
    st.session_state["available_models"] = fetch_openai_models()

# Model selection UI
st.sidebar.header("Model Management")

# Whisper settings in sidebar
st.sidebar.markdown("---")
st.sidebar.header("🎤 Voice Transcription Settings")
use_local = st.sidebar.checkbox(
    "Use Local Whisper Base (FREE)",
    value=st.session_state.use_local_whisper,
    help="Enable to use free local Whisper Base model. Disable to use OpenAI API ($0.006/minute)"
)
if use_local != st.session_state.use_local_whisper:
    st.session_state.use_local_whisper = use_local
    st.rerun()

if st.session_state.use_local_whisper:
    st.sidebar.info("💰 **Cost:** FREE (runs locally)\n📊 **Model:** Whisper Base\n🌍 **Languages:** 99 supported")
else:
    st.sidebar.info("💰 **Cost:** $0.006 per minute\n📊 **Model:** OpenAI Whisper-1\n🌍 **Languages:** Auto-detect")

st.sidebar.markdown("---")
model_names = [m["name"] for m in st.session_state["available_models"]]
def_model = DEFAULT_MODEL_OPTIONS[0]["name"]
selected_model_name = st.sidebar.selectbox("Choose a model", model_names, index=model_names.index(def_model) if def_model in model_names else 0)


# Show model info
selected_model = next((m for m in st.session_state["available_models"] if m["name"] == selected_model_name), None)
if selected_model:
    st.sidebar.markdown(f"**{selected_model['name']}**\n\n{selected_model['description']}")
    # Show pricing table if available
    show_pricing_table(selected_model["id"])
else:
    st.sidebar.warning("Model not found.")

# Show the model being used
st.info(f"Model in use: {selected_model['name']}")

# --- File Uploader - Dynamically set based on selected_model ---
# Determine supported file types for the uploader based on the selected model
if selected_model:
    # Use supported_file_types from model metadata, default to ["txt", "pdf"] if key is missing
    uploader_types = selected_model.get('supported_file_types', ["txt", "pdf"])
    # Create a dynamic key for the file uploader to ensure it resets when types change
    uploader_key = f"file_uploader_{selected_model['id']}"
else:
    # Fallback if selected_model is somehow None (e.g., during initial script issues)
    uploader_types = ["txt", "pdf", "png", "jpg", "jpeg"]  # Broad default
    uploader_key = "file_uploader_default"

uploaded_file = st.file_uploader(
    "Upload a file for AI analysis (types based on selected model)",
    type=uploader_types,
    key=uploader_key
)

# Initialize/reset file processing variables each time after the uploader is rendered
file_content = None
image_base64_data = None
file_type = None

# Process the uploaded file if one exists
# This block needs to be here, after `selected_model` is known, so that image processing
# can correctly determine if the current model is image-capable.
if uploaded_file is not None:
    file_type = uploaded_file.type  # Get the MIME type of the file

    # Process PDF files
    if file_type == "application/pdf":
        try:
            import PyPDF2  # Lazy import for PDF processing
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            # Extract text from all pages and combine
            file_content = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            image_base64_data = None  # Ensure no stale image data
            st.success("PDF uploaded and ready for analysis.")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            file_content = None  # Reset on error

    # Process text files
    elif file_type.startswith("text"):
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        image_base64_data = None  # Ensure no stale image data
        st.success("Text file uploaded and ready for analysis.")

    # Process image files
    elif file_type.startswith("image"):
        # `selected_model` is guaranteed to be defined here due to the placement of this block.
        current_model_id = selected_model['id']

        if current_model_id in IMAGE_CAPABLE_MODELS:
            # For image-capable models, read image bytes and encode to base64
            image_bytes = uploaded_file.getvalue()
            image_base64_data = base64.b64encode(image_bytes).decode()
            file_content = None  # Clear text file content if any
            st.success(f"Image uploaded and ready for analysis with {selected_model['name']}.")
        else:
            # For non-image-capable models, do not process image data.
            file_content = None
            image_base64_data = None
            st.success("Image uploaded. You can ask questions about this file, but image content will not be analyzed by this model.")

    # Handle unsupported file types (though uploader `type` should prevent this)
    else:
        st.warning(f"Unsupported file type: {file_type}. Please upload one of the supported types: {uploader_types}")
        file_content = None
        image_base64_data = None
# --- End of File Uploader and Processing ---


# Button to submit the question (or auto-submit in voice mode)
ask_clicked = False
if not st.session_state.voice_mode:
    ask_clicked = st.button("Ask")
else:
    # In voice mode, automatically process if there's input
    if user_input.strip():
        ask_clicked = True
        st.info("🤖 Processing your voice input automatically...")

if ask_clicked:
    # Check if there is any text input OR if any file content (text or image base64) has been processed
    if user_input.strip() or file_content or image_base64_data:
        with st.spinner('Processing...'):
            try:
                # --- Start of Prompt Construction and API Call Logic ---
                # Check if the selected model is an Ollama model, Claude model, or an OpenAI model
                if not (selected_model["id"] in ["llama3.3", "deepseek-r1", "qwen2.5-coder", "mistral"]):
                    # --- OpenAI Model Path ---
                    messages_payload = [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": []}
                    ]
                    current_content_parts = []  # To build the list of content parts for the prompt

                    user_question = user_input.strip()

                    # Case 1: Image uploaded and model is image-capable
                    if image_base64_data and selected_model["id"] in IMAGE_CAPABLE_MODELS:
                        # Add user's text question as a text part
                        if user_question:
                            current_content_parts.append({"type": "input_text", "text": user_question})
                        else:
                            # Default text if no specific question is asked about the image
                            current_content_parts.append({"type": "input_text", "text": f"Describe this image ({uploaded_file.name})."})

                        # Add image data as an image URL part (base64 encoded)
                        # Use the stored file_type (MIME type) for the data URI. Default to image/jpeg if unknown.
                        mime_type = file_type if file_type and file_type.startswith("image/") else "image/jpeg"
                        current_content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_base64_data}"}
                        })

                    # Case 2: Text or PDF file content is available
                    elif file_content:
                        # Combine file content and user question into a single text part
                        combined_text = f"Analyze the following file content:\n\n{file_content}\n\nUser question: {user_question}"
                        current_content_parts.append({"type": "input_text", "text": combined_text})

                    # Case 3: Image uploaded, but model is NOT image-capable
                    elif file_type and file_type.startswith("image/"):
                        # Inform the user that the image won't be analyzed by this model
                        non_analyzable_prompt = (
                            f"The user uploaded an image file named '{uploaded_file.name}'. "
                            f"Please note: image content will not be analyzed by the current model ({selected_model['name']}). "
                            f"User question: {user_question}"
                        )
                        current_content_parts.append({"type": "input_text", "text": non_analyzable_prompt})

                    # Case 4: Only user text input (no file or unhandled file type)
                    else:
                        if user_question:
                            current_content_parts.append({"type": "input_text", "text": user_question})
                        else:
                            # If no text and no file/image, warn and exit
                            st.warning("Please enter a question or upload a file.")
                            st.stop()

                    messages_payload[0]["content"] = current_content_parts

                # --- Model Execution Logic ---
                if selected_model["id"] in ["llama3.3", "deepseek-r1", "qwen2.5-coder", "mistral"]:
                    # --- Ollama Model Path ---
                    import subprocess  # For running Ollama CLI
                    ollama_model_tag = f"{selected_model['id']}:latest"

                    ollama_prompt_str = f"{SYSTEM_MESSAGE}\n\n"  # Final prompt string for Ollama with system message
                    user_question_for_ollama = user_input.strip()

                    # Check file type to construct the Ollama prompt appropriately
                    if file_type and file_type.startswith("image/"):
                        # Image uploaded, but Ollama models are not treated as image-capable here.
                        # Provide a textual notification about the image.
                        if user_question_for_ollama:
                            ollama_prompt_str = (
                                f"The user uploaded an image file named '{uploaded_file.name}'. "
                                f"This model cannot analyze image content directly. "
                                f"User question: {user_question_for_ollama}"
                            )
                        else:
                            ollama_prompt_str = (
                                f"The user uploaded an image file named '{uploaded_file.name}'. "
                                f"This model cannot analyze image content directly. "
                                f"Please ask a question about the file's context or metadata, or provide a general query."
                            )
                    elif file_content:  # Text or PDF file content
                        if user_question_for_ollama:
                            ollama_prompt_str = (
                                f"Analyze the following file content:\n\n{file_content}\n\n"
                                f"User question: {user_question_for_ollama}"
                            )
                        else:
                            ollama_prompt_str = f"Analyze the following file content:\n\n{file_content}"
                    else:  # Only user text input
                        ollama_prompt_str = user_question_for_ollama

                    # Ensure the final prompt for Ollama is not empty
                    if not ollama_prompt_str.strip():
                        st.warning("Please enter a question or upload a file for the Ollama model.")
                        st.stop()

                    # Execute Ollama command
                    result = subprocess.run(
                        ["ollama", "run", ollama_model_tag],
                        input=ollama_prompt_str,
                        capture_output=True,
                        text=True
                    )
                    output_text = result.stdout.strip() if result.stdout else None
                    if output_text:
                        st.subheader("Model Response:")
                        st.write(output_text)

                        # Add text-to-speech for the response
                        st.markdown("### 🔊 Voice Output")

                        # Escape special characters for JavaScript
                        escaped_text = output_text.replace('`', r'\`').replace('\n', ' ').replace('$', r'\$')

                        # TTS component with built-in controls
                        tts_html = f"""
                        <div id="tts-container" style="padding: 10px; background-color: #000000; border-radius: 8px;">
                            <style>
                                .tts-button {{
                                    background-color: #ff4b4b;
                                    color: white;
                                    border: none;
                                    padding: 10px 20px;
                                    font-size: 16px;
                                    border-radius: 5px;
                                    cursor: pointer;
                                    margin-right: 10px;
                                    transition: all 0.3s;
                                }}
                                .tts-button:hover {{
                                    background-color: #ff3333;
                                }}
                                .tts-button:disabled {{
                                    background-color: #cccccc;
                                    cursor: not-allowed;
                                }}
                                #tts-status {{
                                    margin-top: 10px;
                                    font-size: 14px;
                                    color: #ffffff;
                                }}
                            </style>
                            <button class="tts-button" id="speakBtn" onclick="speakText()">🔊 Read Response</button>
                            <button class="tts-button" id="stopBtn" onclick="stopSpeaking()" disabled>⏹️ Stop Reading</button>
                            <div id="tts-status">Ready to speak</div>
                            <script>
                                function speakText() {{
                                    if ('speechSynthesis' in window) {{
                                        // Cancel any ongoing speech
                                        window.speechSynthesis.cancel();

                                        const text = `{escaped_text}`;
                                        const utterance = new SpeechSynthesisUtterance(text);

                                        utterance.rate = 0.9;
                                        utterance.pitch = 1;
                                        utterance.volume = 1;

                                        utterance.onstart = function() {{
                                            document.getElementById('tts-status').innerHTML = '🔊 Speaking...';
                                            document.getElementById('tts-status').style.color = 'green';
                                            document.getElementById('speakBtn').disabled = true;
                                            document.getElementById('stopBtn').disabled = false;
                                        }};

                                        utterance.onend = function() {{
                                            document.getElementById('tts-status').innerHTML = '✅ Finished speaking';
                                            document.getElementById('tts-status').style.color = 'gray';
                                            document.getElementById('speakBtn').disabled = false;
                                            document.getElementById('stopBtn').disabled = true;
                                        }};

                                        utterance.onerror = function(event) {{
                                            document.getElementById('tts-status').innerHTML = '❌ Error: ' + event.error;
                                            document.getElementById('tts-status').style.color = 'red';
                                            document.getElementById('speakBtn').disabled = false;
                                            document.getElementById('stopBtn').disabled = true;
                                        }};

                                        window.speechSynthesis.speak(utterance);
                                    }} else {{
                                        document.getElementById('tts-status').innerHTML = '❌ Text-to-speech not supported';
                                    }}
                                }}

                                function stopSpeaking() {{
                                    if ('speechSynthesis' in window) {{
                                        window.speechSynthesis.cancel();
                                        document.getElementById('tts-status').innerHTML = '⏹️ Speaking stopped';
                                        document.getElementById('tts-status').style.color = 'orange';
                                        document.getElementById('speakBtn').disabled = false;
                                        document.getElementById('stopBtn').disabled = true;
                                    }}
                                }}
                            </script>
                        </div>
                        """
                        components.html(tts_html, height=120)
                    else:
                        st.write("Sorry, no response from the model.")
                elif selected_model["id"] in ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-sonnet-4-20250514"]:
                    # --- Claude Model API Call ---
                    # Build messages for Anthropic API
                    messages = []

                    # Handle different content types for Claude
                    if file_type and file_type.startswith("image/") and selected_model["id"] in IMAGE_CAPABLE_MODELS:
                        # Image with text
                        content_parts = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": file_type,
                                    "data": image_base64_data
                                }
                            }
                        ]
                        if user_question:
                            content_parts.append({"type": "text", "text": user_question})
                        else:
                            content_parts.append({"type": "text", "text": "What do you see in this image?"})
                        messages.append({"role": "user", "content": content_parts})
                    elif file_content:
                        # Text or PDF file content
                        if user_question:
                            text_content = f"Analyze the following file content:\n\n{file_content}\n\nUser question: {user_question}"
                        else:
                            text_content = f"Analyze the following file content:\n\n{file_content}"
                        messages.append({"role": "user", "content": text_content})
                    else:
                        # Only user text input
                        if user_question:
                            messages.append({"role": "user", "content": user_question})
                        else:
                            st.warning("Please enter a question or upload a file.")
                            st.stop()

                    # Make the API call to Anthropic
                    response = anthropic_client.messages.create(
                        model=selected_model["id"],
                        max_tokens=2048,
                        temperature=0.7,
                        system=SYSTEM_MESSAGE,
                        messages=messages
                    )

                    # Display the model's response
                    output_text = response.content[0].text if response.content else None
                    if output_text:
                        st.subheader("Model Response:")
                        st.write(output_text)

                        # Add text-to-speech for the response
                        st.markdown("### 🔊 Voice Output")

                        # Escape special characters for JavaScript
                        escaped_text = output_text.replace('`', r'\`').replace('\n', ' ').replace('$', r'\$')

                        # TTS component with built-in controls
                        tts_html = f"""
                        <div id="tts-container-claude" style="padding: 10px; background-color: #000000; border-radius: 8px;">
                            <style>
                                .tts-button {{
                                    background-color: #ff4b4b;
                                    color: white;
                                    border: none;
                                    padding: 10px 20px;
                                    font-size: 16px;
                                    border-radius: 5px;
                                    cursor: pointer;
                                    margin-right: 10px;
                                    transition: all 0.3s;
                                }}
                                .tts-button:hover {{
                                    background-color: #ff3333;
                                }}
                                .tts-button:disabled {{
                                    background-color: #cccccc;
                                    cursor: not-allowed;
                                }}
                                #tts-status-claude {{
                                    margin-top: 10px;
                                    font-size: 14px;
                                    color: #ffffff;
                                }}
                            </style>
                            <button class="tts-button" id="speakBtnClaude" onclick="speakTextClaude()">🔊 Read Response</button>
                            <button class="tts-button" id="stopBtnClaude" onclick="stopSpeakingClaude()" disabled>⏹️ Stop Reading</button>
                            <div id="tts-status-claude">Ready to speak</div>
                            <script>
                                function speakTextClaude() {{
                                    if ('speechSynthesis' in window) {{
                                        // Cancel any ongoing speech
                                        window.speechSynthesis.cancel();

                                        const text = `{escaped_text}`;
                                        const utterance = new SpeechSynthesisUtterance(text);

                                        utterance.rate = 0.9;
                                        utterance.pitch = 1;
                                        utterance.volume = 1;

                                        utterance.onstart = function() {{
                                            document.getElementById('tts-status-claude').innerHTML = '🔊 Speaking...';
                                            document.getElementById('tts-status-claude').style.color = 'green';
                                            document.getElementById('speakBtnClaude').disabled = true;
                                            document.getElementById('stopBtnClaude').disabled = false;
                                        }};

                                        utterance.onend = function() {{
                                            document.getElementById('tts-status-claude').innerHTML = '✅ Finished speaking';
                                            document.getElementById('tts-status-claude').style.color = 'gray';
                                            document.getElementById('speakBtnClaude').disabled = false;
                                            document.getElementById('stopBtnClaude').disabled = true;
                                        }};

                                        utterance.onerror = function(event) {{
                                            document.getElementById('tts-status-claude').innerHTML = '❌ Error: ' + event.error;
                                            document.getElementById('tts-status-claude').style.color = 'red';
                                            document.getElementById('speakBtnClaude').disabled = false;
                                            document.getElementById('stopBtnClaude').disabled = true;
                                        }};

                                        window.speechSynthesis.speak(utterance);
                                    }} else {{
                                        document.getElementById('tts-status-claude').innerHTML = '❌ Text-to-speech not supported';
                                    }}
                                }}

                                function stopSpeakingClaude() {{
                                    if ('speechSynthesis' in window) {{
                                        window.speechSynthesis.cancel();
                                        document.getElementById('tts-status-claude').innerHTML = '⏹️ Speaking stopped';
                                        document.getElementById('tts-status-claude').style.color = 'orange';
                                        document.getElementById('speakBtnClaude').disabled = false;
                                        document.getElementById('stopBtnClaude').disabled = true;
                                    }}
                                }}
                            </script>
                        </div>
                        """
                        components.html(tts_html, height=120)
                    else:
                        st.write("Sorry, no response from the model.")
                else:
                    # --- OpenAI Model API Call ---
                    # Ensure there's content to send
                    if not messages_payload[0]["content"]:
                        st.warning("Please enter a question to accompany the uploaded file.")
                        st.stop()

                    # Make the API call to OpenAI
                    # Note: client.responses.create seems to be a placeholder or custom SDK method.
                    # Standard OpenAI SDK uses client.chat.completions.create with a 'messages' parameter.
                    # This code assumes client.responses.create is adapted for this payload.
                    response = client.responses.create(
                        model=selected_model["id"],
                        input=messages_payload,  # Pass the structured multimodal payload
                        # The 'text' parameter (used in previous versions) is removed,
                        # as it's typically not used for multimodal chat completion calls.
                        reasoning={},
                        tools=[],
                        temperature=0.7,
                        max_output_tokens=2048,
                        top_p=1,
                        store=True  # Assuming this is a valid parameter for the client
                    )
                    # Display the model's response
                    # Parse the output to extract the text
                    output_text = None
                    if hasattr(response, 'output') and response.output:
                        if isinstance(response.output, list) and len(response.output) > 0:
                            first = response.output[0]
                            if hasattr(first, 'content') and isinstance(first.content, list) and len(first.content) > 0:
                                content_item = first.content[0]
                                if hasattr(content_item, 'text'):
                                    output_text = content_item.text
                                elif isinstance(content_item, dict) and 'text' in content_item:
                                    output_text = content_item['text']
                            elif hasattr(first, 'text'):
                                output_text = first.text
                            elif isinstance(first, dict) and 'text' in first:
                                output_text = first['text']
                            else:
                                output_text = str(first)
                        else:
                            output_text = str(response.output)
                        st.subheader("Model Response:")
                        st.write(output_text)

                        # Add text-to-speech for the response
                        st.markdown("### 🔊 Voice Output")

                        # Escape special characters for JavaScript
                        escaped_text = output_text.replace('`', r'\`').replace('\n', ' ').replace('$', r'\$')

                        # TTS component with built-in controls
                        tts_html = f"""
                        <div id="tts-container-openai" style="padding: 10px; background-color: #000000; border-radius: 8px;">
                            <style>
                                .tts-button {{
                                    background-color: #ff4b4b;
                                    color: white;
                                    border: none;
                                    padding: 10px 20px;
                                    font-size: 16px;
                                    border-radius: 5px;
                                    cursor: pointer;
                                    margin-right: 10px;
                                    transition: all 0.3s;
                                }}
                                .tts-button:hover {{
                                    background-color: #ff3333;
                                }}
                                .tts-button:disabled {{
                                    background-color: #cccccc;
                                    cursor: not-allowed;
                                }}
                                #tts-status-openai {{
                                    margin-top: 10px;
                                    font-size: 14px;
                                    color: #ffffff;
                                }}
                            </style>
                            <button class="tts-button" id="speakBtnOpenAI" onclick="speakTextOpenAI()">🔊 Read Response</button>
                            <button class="tts-button" id="stopBtnOpenAI" onclick="stopSpeakingOpenAI()" disabled>⏹️ Stop Reading</button>
                            <div id="tts-status-openai">Ready to speak</div>
                            <script>
                                function speakTextOpenAI() {{
                                    if ('speechSynthesis' in window) {{
                                        // Cancel any ongoing speech
                                        window.speechSynthesis.cancel();

                                        const text = `{escaped_text}`;
                                        const utterance = new SpeechSynthesisUtterance(text);

                                        utterance.rate = 0.9;
                                        utterance.pitch = 1;
                                        utterance.volume = 1;

                                        utterance.onstart = function() {{
                                            document.getElementById('tts-status-openai').innerHTML = '🔊 Speaking...';
                                            document.getElementById('tts-status-openai').style.color = 'green';
                                            document.getElementById('speakBtnOpenAI').disabled = true;
                                            document.getElementById('stopBtnOpenAI').disabled = false;
                                        }};

                                        utterance.onend = function() {{
                                            document.getElementById('tts-status-openai').innerHTML = '✅ Finished speaking';
                                            document.getElementById('tts-status-openai').style.color = 'gray';
                                            document.getElementById('speakBtnOpenAI').disabled = false;
                                            document.getElementById('stopBtnOpenAI').disabled = true;
                                        }};

                                        utterance.onerror = function(event) {{
                                            document.getElementById('tts-status-openai').innerHTML = '❌ Error: ' + event.error;
                                            document.getElementById('tts-status-openai').style.color = 'red';
                                            document.getElementById('speakBtnOpenAI').disabled = false;
                                            document.getElementById('stopBtnOpenAI').disabled = true;
                                        }};

                                        window.speechSynthesis.speak(utterance);
                                    }} else {{
                                        document.getElementById('tts-status-openai').innerHTML = '❌ Text-to-speech not supported';
                                    }}
                                }}

                                function stopSpeakingOpenAI() {{
                                    if ('speechSynthesis' in window) {{
                                        window.speechSynthesis.cancel();
                                        document.getElementById('tts-status-openai').innerHTML = '⏹️ Speaking stopped';
                                        document.getElementById('tts-status-openai').style.color = 'orange';
                                        document.getElementById('speakBtnOpenAI').disabled = false;
                                        document.getElementById('stopBtnOpenAI').disabled = true;
                                    }}
                                }}
                            </script>
                        </div>
                        """
                        components.html(tts_html, height=120)
                    else:
                        st.write("Sorry, no response from the model.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question or upload a file.")
