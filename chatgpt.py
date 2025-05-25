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
                if hasattr(response, 'output') and response.output:
                    st.subheader("Model Response:")
                    st.write(response.output)
                else:
                    st.write("Sorry, no response from the model.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")
