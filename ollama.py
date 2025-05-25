import streamlit as st
import subprocess
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
    return "Model Loaded"


# Load the model (this will run only once per session)
model_status = load_model()

# Button to submit the question
if st.button("Ask"):
    if user_input.strip():
        # Show a spinner while processing the model request
        with st.spinner('Processing...'):
            try:
                # Run the ollama command and pass the user input via stdin
                result = subprocess.run(
                    ["ollama", "run", "llama3:latest"],  # Correct model name
                    input=user_input,                    # Provide input directly via stdin
                    capture_output=True,
                    text=True
                )

                # Check if there's output from the model
                if result.stdout.strip():
                    st.subheader("Model Response:")
                    st.write(result.stdout.strip())  # Display the model's response
                else:
                    st.write("Sorry, no response from the model.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")
