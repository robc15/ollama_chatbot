import streamlit as st
from openai import OpenAI
import time

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

# Show the model being used
st.info("Model in use: gpt-4.1")

# Button to submit the question
if st.button("Ask"):
    if user_input.strip():
        with st.spinner('Processing...'):
            try:
                response = client.responses.create(
                    model="gpt-4.1",
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
