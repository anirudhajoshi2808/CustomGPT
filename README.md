# Prompt Generator for Data Science Engg Tools and Methods

This app generates responses based on prompts related to O'Reilly's "Introduction to Machine Learning with Python" book using the GPT-2 language model. Contains jupyter notebook with model implementation and streamlit app for user interface. It has 4 conversation starters and followup prompts. 

![img](https://i.imgur.com/wTb7Zct.png)


## Getting Started

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:

    ```bash
    streamlit run streamlit-app.py
    ```

3. Access the app in your web browser at the provided URL.

## File Structure

- **streamlit-app.py**: The main Streamlit application script.
- **requirements.txt**: A list of Python packages required for the app.
- **CustomGPT_DSEMT.ipynb**: Jupyter Notebook containing the actual model implementation. It covers steps such as converting PDF to text, cleaning the dataset, and model training.

## Model Implementation

The core implementation of the model can be found in the `implementation.ipynb` Jupyter Notebook. It covers the following steps:

- Converting the PDF of "Introduction to Machine Learning with Python" to text.
- Cleaning the dataset, including removing punctuation and stop words.
- Training the GPT-2 language model on the preprocessed text.
- Generating responses based on user prompts.

Feel free to explore the notebook for a detailed walkthrough of the model implementation.

## Usage

- The app provides a dropdown menu to select prompts related to the book.
- The generated responses based on the selected prompt are displayed on the web page.
- The cover photo of the book is included for visual appeal.

## Customization

- You can customize the prompts in the `book_prompts` list in the `app.py` file.
- Replace the placeholder cover photo path with the actual path in the `st.image` function.

## Dependencies

- [Streamlit](https://www.streamlit.io/)
- [Transformers](https://huggingface.co/transformers)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

## Acknowledgments

- This app uses the GPT-2 language model provided by the Transformers library.
- Cover photo courtesy of O'Reilly Media.

