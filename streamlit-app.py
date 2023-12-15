import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

prompts = [
    "What is overfitting?",
    "What is underfitting?",
    "What are the advantages of using neural networks for image recognition?",
    "Apply the RBF kernel SVM to the Breast Cancer dataset"
]

def generate_response(prompt, num_responses=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=num_responses,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        use_cache=True,
        num_beams=5,
    )

    decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    return decoded_output

def main():
    st.title("Prompt Generator for Data Science Engg Tools and Methods")
    st.write("This Streamlit app generates responses based on prompts related to O'Reilly's Introduction to Machine Learning with Python book.")
    st.image("book.png", caption="Cover Photo", use_column_width=True)

    selected_starter = st.selectbox("Select a Conversation Starter", prompts)

    starter_responses = generate_response(selected_starter, num_responses=1)
    st.write("Model: {}".format(starter_responses[0]))

    if st.button("Why do you think that?"):
        follow_up_responses = generate_response("Why do you think that?", num_responses=1)
        st.text("Model: {}".format(follow_up_responses[0]))

    if st.button("Be more specific"):
        follow_up_responses = generate_response("Be more specific", num_responses=1)
        st.text("Model: {}".format(follow_up_responses[0]))

if __name__ == "__main__":
    main()
