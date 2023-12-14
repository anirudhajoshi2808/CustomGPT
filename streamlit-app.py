import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import fitz  # PyMuPDF

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# List of prompts
prompts = [
    "Apply the RBF kernel SVM to the Breast Cancer dataset",
    "What is overfitting?",
    "What are the advantages of using neural networks for image recognition?",
    # Add more prompts as needed
]

# Function to generate responses based on the selected prompt
def generate_responses(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        use_cache=True,
        num_beams=5,
    )

    decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    return decoded_output

# Streamlit app
st.title("Prompt Generator for Data Science Engg Tools and Methods")
st.write("This Streamlit app generates responses based on prompts related to O'Reilly's Introduction to Machine Learning with Python book.")
st.image("book.png", caption="Cover Photo", use_column_width=True)


# Dropdown for selecting prompts
selected_prompt = st.selectbox("Select a prompt:", prompts)

# Generate and display responses
generated_responses = generate_responses(selected_prompt)

st.markdown(f"**Prompt:** {selected_prompt}")
st.markdown("**Generated Responses:**")
for response in generated_responses:
    st.write(response)
    st.markdown("---")


# st.title("Prompt Generator for Data Science Engg Tools and Methods")
# st.image("book.png", caption="Cover Photo", use_column_width=True)

# with st.container():
    
#     st.write("This Streamlit app generates responses based on prompts related to O'Reilly's Introduction to Machine Learning with Python book.")

#     st.header("Select a Prompt")
    
#     selected_prompt = st.selectbox("Choose a prompt:", prompts)

#     generated_responses = generate_responses(selected_prompt)

#     st.subheader(f"Selected Prompt: {selected_prompt}")
#     st.subheader("Generated Responses:")
#     for response in generated_responses:
#         st.write(response)
#         st.markdown("---")
