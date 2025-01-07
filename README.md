# FAQ Chatbot with Flexible CSV Column Names

This is an interactive FAQ chatbot built with **Streamlit** that allows users to upload a CSV file containing questions and answers. Users can ask questions, and the chatbot will find and display the most relevant answer based on text similarity. 

## Features
- **Flexible CSV Handling**: Automatically detects columns for questions and answers, even if they have different names.
- **Similarity Models**: 
  - Uses **TF-IDF** for text similarity.
  - Can be extended to include advanced models like BERT.
- **Dynamic Similarity Threshold**: Allows users to adjust the similarity threshold for better results.
- **Sidebar FAQ Display**: Lists all questions from the FAQ dataset for quick reference.
- **Error Handling**: Ensures the uploaded CSV file is valid and correctly formatted.

---

## Requirements

### Python Libraries
The following libraries are required to run the program:
- `streamlit`
- `pandas`
- `transformers`
- `scikit-learn`
- `nltk`
- `numpy`

### Install the Dependencies
Run the following command to install all required libraries:
```bash
pip install streamlit pandas transformers scikit-learn nltk numpy
```

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/PratikMitkar/codealpha_tasks_2.git
cd codealpha_tasks_2
```

### 2. Run the Program
Start the Streamlit app by running:
```bash
streamlit run faq_chatbot_assignment_2.py
```

### 3. Upload the FAQ CSV File
- The file should contain two columns: one for questions and one for answers.
- Supported column names (case-insensitive):
  - For questions: `question`, `questions`
  - For answers: `answer`, `answers`

### 4. Ask a Question
- Type your question in the input box.
- The chatbot will display the most relevant answer based on the selected similarity model.

---

## CSV File Format

Ensure your CSV file has columns for questions and answers. Example:

| Question                   | Answer                                  |
|----------------------------|-----------------------------------------|
| What is your name?         | My name is FAQ Bot.                    |
| How does the chatbot work? | It finds similar questions using TF-IDF.|

---

## Advanced Features
- **Preprocessing**:
  - Removes stop words and punctuation for better text matching.
- **Similarity Models**:
  - Currently supports **TF-IDF**.
  - Easily extendable to advanced models like BERT.
- **Caching**:
  - Transformer models and similarity computations are cached for improved performance.

---

## Customization

### Add More Similarity Models
To integrate more models:
1. Add your model to the `model_choice` dropdown:
   ```python
   model_choice = st.selectbox(
       "Choose the similarity model:",
       ["TF-IDF", "BERT"]  # Add more options here
   )
   ```
2. Implement a function for the new model and integrate it into the app logic.

### Change Similarity Threshold
Adjust the default similarity threshold in the slider:
```python
similarity_threshold = st.slider("Set similarity threshold:", 0.0, 1.0, 0.2)
```

---

## Known Issues
- Limited to text-based questions and answers.
- Requires a valid CSV format with appropriate columns.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for new features, bug fixes, or documentation improvements.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Built using [Streamlit](https://streamlit.io/).
- Utilizes pre-trained models from [Hugging Face Transformers](https://huggingface.co/transformers/).
- Inspired by modern FAQ systems and NLP techniques.
