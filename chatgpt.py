import streamlit as st
from openai import OpenAI
import time
import datetime
import os
import requests

# Force Streamlit to run on port 8502
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

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
        "default": True
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "description": (
            "OpenAI's new Omni model. Multimodal (text, vision, audio), extremely fast, and cost-effective. "
            "Great for real-time and broad applications."
        ),
        "cost": "$5.00 / 1M input tokens, $15.00 / 1M output tokens",
        "default": False
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "description": (
            "A lighter, faster, and even more affordable version of GPT-4o. "
            "Ideal for high-volume, low-latency tasks where cost is critical."
        ),
        "cost": "$1.00 / 1M input tokens, $2.00 / 1M output tokens",
        "default": False
    },
    {
        "id": "llama3",
        "name": "Llama 3",
        "description": (
            "Meta's Llama 3 model, running locally via Olloma. Great for privacy, offline use, and fast responses on supported hardware. "
            "Best for general chat, coding, and experimentation without cloud costs."
        ),
        "cost": "$0 (runs locally via Olloma)",
        "default": False
    }
]


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
        # Always add Llama 3 (Olloma) as an option
        if not any(m["id"] == "llama3" for m in available):
            llama3_meta = next((m for m in DEFAULT_MODEL_OPTIONS if m["id"] == "llama3"), None)
            if llama3_meta:
                available.append(llama3_meta)
        return available
    except Exception as e:
        st.warning(f"Could not fetch models from OpenAI: {e}")
        # Always add Llama 3 (Olloma) as an option
        fallback = DEFAULT_MODEL_OPTIONS.copy()
        if not any(m["id"] == "llama3" for m in fallback):
            fallback.append({
                "id": "llama3",
                "name": "Llama 3",
                "description": (
                    "Meta's Llama 3 model, running locally via Olloma. Great for privacy, offline use, and fast responses on supported hardware. "
                    "Best for general chat, coding, and experimentation without cloud costs."
                ),
                "cost": "$0 (runs locally via Olloma)",
                "default": False
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
    ]
}


def show_pricing_table(model_id):
    if model_id in PRICING_TABLES:
        st.sidebar.markdown("**Pricing (per 1M tokens):**")
        st.sidebar.table(
            {
                "Type": [row[0] for row in PRICING_TABLES[model_id]],
                "Price": [row[1] for row in PRICING_TABLES[model_id]],
            }
        )


# Show model info and pricing table
selected_model = next((m for m in st.session_state["available_models"] if m["name"] == selected_model_name), None)
if selected_model:
    st.sidebar.markdown(f"**{selected_model['name']}**\n\n{selected_model['description']}")
    # Show pricing table if available
    show_pricing_table(selected_model["id"])
else:
    st.sidebar.warning("Model not found.")

# Show the model being used
st.info(f"Model in use: {selected_model['name']}")


# Helper to fetch and sum paginated usage/costs
def fetch_organization_metric(url, headers, value_key=None):
    total = 0
    total_input_tokens = 0
    total_output_tokens = 0
    while url:
        resp = requests.get(url, headers=headers)
        if not resp.ok:
            break
        data = resp.json()
        for bucket in data.get("data", []):
            for result in bucket.get("results", []):
                if value_key == "cost":
                    # For costs, sum result["amount"]["value"]
                    amount = result.get("amount", {}).get("value", 0)
                    total += amount
                elif value_key == "completions":
                    # For completions, sum input_tokens and output_tokens
                    total_input_tokens += result.get("input_tokens", 0)
                    total_output_tokens += result.get("output_tokens", 0)
                else:
                    # For other usage, sum result[value_key]
                    total += result.get(value_key, 0)
        # Pagination: next_page is a string token, not a full URL
        next_page = data.get("next_page")
        if next_page:
            url = url.split("?")[0] + "?page=" + next_page
        else:
            url = None
    if value_key == "completions":
        return total_input_tokens, total_output_tokens
    return total


# Usage in your try block:
try:
    # Always use OPENAI_ADMIN_KEY for usage/costs API calls
    api_key = os.getenv("OPENAI_ADMIN_KEY")
    if not api_key:
        st.sidebar.warning("OPENAI_ADMIN_KEY environment variable not set.")
    else:
        headers = {"Authorization": f"Bearer {api_key}"}
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=30)  # last 30 days
        end_date = today

        start_time = int(time.mktime(start_date.timetuple()))
        end_time = int(time.mktime((end_date + datetime.timedelta(days=1)).timetuple()))

        # Completions usage (sum input/output tokens)
        completions_url = f"https://api.openai.com/v1/organization/usage/completions?start_time={start_time}&end_time={end_time}"
        input_tokens, output_tokens = fetch_organization_metric(completions_url, headers, value_key="completions")

        # Costs
        costs_url = f"https://api.openai.com/v1/organization/costs?start_time={start_time}&end_time={end_time}"
        total_cost = fetch_organization_metric(costs_url, headers, value_key="cost")  # cost is in dollars

        st.sidebar.markdown("---")
        if input_tokens > 0 or output_tokens > 0:
            st.sidebar.info(
                f"**Completions Usage (last 30 days):**\n"
                f"- Input tokens: {input_tokens:,}\n"
                f"- Output tokens: {output_tokens:,}"
            )
        else:
            st.sidebar.info("**Completions Usage:** Not available.")
        if total_cost > 0:
            st.sidebar.success(f"**Total Cost (last 30 days):** ${total_cost:,.2f}")
        else:
            st.sidebar.info("**Total Cost:** Not available.")

except Exception as e:
    st.sidebar.warning(f"Could not fetch usage/costs: {e}")

# Button to submit the question
if st.button("Ask"):
    if user_input.strip():
        with st.spinner('Processing...'):
            try:
                if selected_model["id"] == "llama3":
                    import subprocess
                    result = subprocess.run(
                        ["olloma", "run", "llama3:latest"],
                        input=user_input,
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
                    response = client.responses.create(
                        model=selected_model["id"],
                        input=[{"role": "user", "content": user_input}],
                        text={
                            "format": {
                                "type": "text"
                            }
                        },
                        reasoning={},
                        tools=[],
                        temperature=1,
                        max_output_tokens=2048,
                        top_p=1,
                        store=True
                    )
                    # Display the model's response
                    # Parse the output to extract the text
                    output_text = None
                    if hasattr(response, 'output') and response.output:
                        # If output is a list of objects, extract the text from the first one
                        if isinstance(response.output, list) and len(response.output) > 0:
                            first = response.output[0]
                            # Try to extract text from known structure
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
        st.warning("Please enter a question.")
