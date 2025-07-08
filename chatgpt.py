import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import time
import os
import base64

# Force Streamlit to run on port 8502
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

# Image capable models: A list of model IDs that are known to support image input (multimodal).
# This list is used to determine if an uploaded image should be processed into base64 data
# and sent to the model in a multimodal format.
IMAGE_CAPABLE_MODELS = ['gpt-4o', 'gpt-4o-mini']

# System message for better AI responses
SYSTEM_MESSAGE = "You are a helpful, concise assistant that speaks clearly and answers with expertise. Provide accurate, well-structured responses that directly address the user's question."

# Title of the app
st.title("AI Robot 🤖")

# Description of the app
st.write("Enter a question below and get a response from the model!")

# Text input for the user question
user_input = st.text_area("Your Question", "", height=150)


# Cache the model load process to optimize performance (only run once per session)
@st.cache_resource
def load_model():
    # Simulate model loading time (you can replace this with actual model loading if needed)
    time.sleep(2)  # Simulating the model loading time
    return OpenAI()


# Load the model (this will run only once per session)
client = load_model()

# Initialize Anthropic client
@st.cache_resource
def load_anthropic_client():
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    if not claude_api_key:
        st.warning("CLAUDE_API_KEY environment variable not set. Claude models will not be available.")
        return None
    return Anthropic(api_key=claude_api_key)

anthropic_client = load_anthropic_client()

# Model options with descriptions and cost per usage
DEFAULT_MODEL_OPTIONS = [
    {
        "id": "gpt-4.1",
        "name": "GPT-4.1",
        "description": (
            "OpenAI's flagship model. Best for advanced reasoning, creativity, and complex tasks. "
            "Excels at coding, analysis, and nuanced conversation."
        ),
        "cost": "$10.00 / 1M input tokens, $30.00 / 1M output tokens",
        "default": True,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "description": (
            "OpenAI's new Omni model. Multimodal (text, vision, audio), extremely fast, and cost-effective. "
            "Great for real-time and broad applications."
        ),
        "cost": "$5.00 / 1M input tokens, $15.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "description": (
            "A lighter, faster, and even more affordable version of GPT-4o. "
            "Ideal for high-volume, low-latency tasks where cost is critical."
        ),
        "cost": "$1.00 / 1M input tokens, $2.00 / 1M output tokens",
        "default": False,
        "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
    },
    {
        "id": "llama3",
        "name": "Llama 3",
        "description": (
            "Meta's Llama 3 model, running locally via Ollama. Great for privacy, offline use, and fast responses on supported hardware. "
            "Best for general chat, coding, and experimentation without cloud costs."
        ),
        "cost": "$0 (runs locally via Ollama)",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "gemma",
        "name": "Gemma",
        "description": (
            "Google's Gemma model, running locally via Ollama. Great for privacy, offline use, and fast responses on supported hardware. "
            "Best for general chat, coding, and experimentation without cloud costs."
        ),
        "cost": "$0 (runs locally via Ollama)",
        "default": False,
        "supported_file_types": ["txt", "pdf"]
    },
    {
        "id": "deepseek-r1",
        "name": "DeepSeek",
        "description": (
            "A powerful model from DeepSeek AI, known for its strong coding and reasoning capabilities. "
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
    "llama3": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ],
    "gemma": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ],
    "deepseek-r1": [
        ["Input", "$0 (runs locally)"],
        ["Output", "$0 (runs locally)"]
    ],
    "claude-3-7-sonnet-20250219": [
        ["Input", "$3.00"],
        ["Output", "$15.00"]
    ],
    "claude-sonnet-4-20250514": [
        ["Input", "$15.00"],
        ["Output", "$75.00"]
    ],
    "claude-3-5-haiku-20241022": [
        ["Input", "$1.00"],
        ["Output", "$5.00"]
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
        # Always add Llama 3 (Ollama) as an option
        if not any(m["id"] == "llama3" for m in available):
            llama3_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "llama3"), None)
            if llama3_meta:
                available.append(llama3_meta)
        # Always add Gemma (Ollama) as an option
        if not any(m["id"] == "gemma" for m in available):
            gemma_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "gemma"), None)
            if gemma_meta:
                available.append(gemma_meta)
        # Always add DeepSeek (Ollama) as an option
        if not any(m["id"] == "deepseek-r1" for m in available):
            deepseek_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "deepseek-r1"), None)
            if deepseek_meta:
                available.append(deepseek_meta)
        # Always add Claude models as options if API key is available
        if anthropic_client and not any(m["id"] == "claude-3-7-sonnet-20250219" for m in available):
            claude_37_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "claude-3-7-sonnet-20250219"), None)
            if claude_37_meta:
                available.append(claude_37_meta)
        if anthropic_client and not any(m["id"] == "claude-sonnet-4-20250514" for m in available):
            claude_4_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "claude-sonnet-4-20250514"), None)
            if claude_4_meta:
                available.append(claude_4_meta)
        if anthropic_client and not any(m["id"] == "claude-3-5-haiku-20241022" for m in available):
            claude_haiku_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "claude-3-5-haiku-20241022"), None)
            if claude_haiku_meta:
                available.append(claude_haiku_meta)
        return available
    except Exception as e:
        st.warning(f"Could not fetch models from OpenAI: {e}")
        # Always add Llama 3 (Ollama) as an option
        fallback = DEFAULT_MODEL_OPTIONS.copy()
        if not any(m["id"] == "llama3" for m in fallback):
            fallback.append({
                "id": "llama3",
                "name": "Llama 3",
                "description": (
                    "Meta's Llama 3 model, running locally via Ollama. Great for privacy, offline use, and fast responses on supported hardware. "
                    "Best for general chat, coding, and experimentation without cloud costs."
                ),
                "cost": "$0 (runs locally via Ollama)",
                "default": False,
                "supported_file_types": ["txt", "pdf"]
            })
        # Always add Gemma (Ollama) as an option
        if not any(m["id"] == "gemma" for m in fallback):
            fallback.append({
                "id": "gemma",
                "name": "Gemma",
                "description": (
                    "Google's Gemma model, running locally via Ollama. Great for privacy, offline use, and fast responses on supported hardware. "
                    "Best for general chat, coding, and experimentation without cloud costs."
                ),
                "cost": "$0 (runs locally via Ollama)",
                "default": False,
                "supported_file_types": ["txt", "pdf"]
            })
        # Always add DeepSeek (Ollama) as an option
        if not any(m["id"] == "deepseek-r1" for m in fallback):
            fallback.append({
                "id": "deepseek-r1",
                "name": "DeepSeek",
                "description": (
                    "A powerful model from DeepSeek AI, known for its strong coding and reasoning capabilities. "
                    "Runs locally via Ollama."
                ),
                "cost": "$0 (runs locally via Ollama)",
                "default": False,
                "supported_file_types": ["txt", "pdf"]
            })
        # Always add Claude models as options if API key is available
        if anthropic_client and not any(m["id"] == "claude-3-7-sonnet-20250219" for m in fallback):
            fallback.append({
                "id": "claude-3-7-sonnet-20250219",
                "name": "Claude 3.7 Sonnet",
                "description": (
                    "Anthropic's Claude 3.7 Sonnet model with enhanced capabilities. Superior at writing, analysis, coding, and complex reasoning tasks. "
                    "Advanced vision support with excellent image understanding and document analysis."
                ),
                "cost": "$3.00 / 1M input tokens, $15.00 / 1M output tokens",
                "default": False,
                "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
            })
        if anthropic_client and not any(m["id"] == "claude-sonnet-4-20250514" for m in fallback):
            fallback.append({
                "id": "claude-sonnet-4-20250514",
                "name": "Claude 4.0 Sonnet",
                "description": (
                    "Anthropic's latest flagship model with superior performance. Exceptional at reasoning, analysis, coding, and creative tasks. "
                    "Advanced multimodal capabilities with cutting-edge image understanding and document analysis."
                ),
                "cost": "$15.00 / 1M input tokens, $75.00 / 1M output tokens",
                "default": False,
                "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
            })
        if anthropic_client and not any(m["id"] == "claude-3-5-haiku-20241022" for m in fallback):
            fallback.append({
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "description": (
                    "Anthropic's fastest model with enhanced capabilities. Optimized for speed and efficiency with improved reasoning. "
                    "Great for quick responses, coding assistance, and high-volume use cases while being cost-effective."
                ),
                "cost": "$1.00 / 1M input tokens, $5.00 / 1M output tokens",
                "default": False,
                "supported_file_types": ["txt", "pdf", "png", "jpg", "jpeg"]
            })
        return fallback


# Session state for models (to allow deletion)
if "available_models" not in st.session_state:
    st.session_state["available_models"] = fetch_openai_models()

# Model selection UI
st.sidebar.header("Model Management")
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


# Button to submit the question
if st.button("Ask"):
    # Check if there is any text input OR if any file content (text or image base64) has been processed
    if user_input.strip() or file_content or image_base64_data:
        with st.spinner('Processing...'):
            try:
                # --- Start of Prompt Construction and API Call Logic ---
                # Check if the selected model is an Ollama model, Claude model, or an OpenAI model
                if False:  # Claude models temporarily removed
                    # --- Claude Model Path ---
                    if not anthropic_client:
                        st.error("Claude API key not configured. Please set the CLAUDE_API_KEY environment variable.")
                        st.stop()
                    
                    messages_payload = []
                    user_question = user_input.strip()
                    
                    # Case 1: Image uploaded and model is image-capable
                    if image_base64_data:
                        content_parts = []
                        
                        # Add user's text question
                        if user_question:
                            content_parts.append({
                                "type": "text",
                                "text": user_question
                            })
                        else:
                            content_parts.append({
                                "type": "text",
                                "text": f"Describe this image ({uploaded_file.name})."
                            })
                        
                        # Add image data
                        mime_type = file_type if file_type and file_type.startswith("image/") else "image/jpeg"
                        
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_base64_data
                            }
                        })
                        
                        messages_payload.append({
                            "role": "user",
                            "content": content_parts
                        })
                    
                    # Case 2: Text or PDF file content is available
                    elif file_content:
                        combined_text = f"Analyze the following file content:\n\n{file_content}\n\nUser question: {user_question}"
                        messages_payload.append({
                            "role": "user",
                            "content": combined_text
                        })
                    
                    # Case 3: Only user text input
                    else:
                        if user_question:
                            messages_payload.append({
                                "role": "user",
                                "content": user_question
                            })
                        else:
                            st.warning("Please enter a question or upload a file.")
                            st.stop()
                    
                    # Make the API call to Claude
                    response = anthropic_client.messages.create(
                        model=selected_model["id"],
                        max_tokens=2048,
                        temperature=0.7,
                        system=SYSTEM_MESSAGE,
                        messages=messages_payload
                    )
                    
                    # Display the response
                    if response.content and len(response.content) > 0:
                        output_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                        st.subheader("Model Response:")
                        st.write(output_text)
                    else:
                        st.write("Sorry, no response from the model.")
                
                elif not (selected_model["id"] in ["llama3", "gemma", "deepseek-r1"]):
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
                if selected_model["id"] in ["llama3", "gemma", "deepseek-r1"]:
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
                    else:
                        st.write("Sorry, no response from the model.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question or upload a file.")
