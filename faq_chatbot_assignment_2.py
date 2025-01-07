import streamlit as st
import pandas as pd
from transformers import pipeline, AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Preprocessing for TF-IDF
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stop_words])

# Load transformer pipeline with a more advanced model
@st.cache_resource
def load_transformer_model():
    model_name = "bert-base-uncased"  # Improved model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


# Compute similarity with TF-IDF (add caching)
@st.cache_data
def get_tfidf_similarity(user_question, faqs):
    questions = faqs["Question"].tolist()
    preprocessed_questions = [preprocess_text(q) for q in questions]
    preprocessed_user_question = preprocess_text(user_question)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_user_question])
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    
    best_match_idx = np.argmax(similarity_scores)
    return faqs.iloc[best_match_idx]["Answer"], similarity_scores[best_match_idx]

# Helper function to find valid columns
def find_columns(df, possible_question_names, possible_answer_names):
    question_col = None
    answer_col = None

    for col in df.columns:
        if col.lower() in [q.lower() for q in possible_question_names]:
            question_col = col
        if col.lower() in [a.lower() for a in possible_answer_names]:
            answer_col = col

    return question_col, answer_col

# Streamlit App
st.title("FAQ Chatbot with Flexible CSV Column Names")
st.write("Ask a question, and the chatbot will answer using the uploaded FAQ dataset.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your FAQs CSV file", type=["csv"])

if uploaded_file:
    try:
        faqs = pd.read_csv(uploaded_file)

        # Define possible column names
        possible_question_names = ["question", "questions"]
        possible_answer_names = ["answer", "answers"]

        # Detect columns
        question_col, answer_col = find_columns(faqs, possible_question_names, possible_answer_names)

        if question_col and answer_col:
            faqs = faqs.rename(columns={question_col: "Question", answer_col: "Answer"})
            st.success("FAQs loaded successfully!")

            # Model selection
            model_choice = st.selectbox(
                "Choose the similarity model:",
                ["TF-IDF"]
            )
            
            # User question input
            user_question = st.text_input("Your question:")

            if user_question:
                with st.spinner("Processing your question..."):
                    if model_choice == "TF-IDF":
                        answer, score = get_tfidf_similarity(user_question, faqs)
                    else:
                        answer, score = "Invalid model choice.", 0

                    # Add a slider for similarity threshold
                    similarity_threshold = st.slider("Set similarity threshold:", 0.0, 1.0, 0.2)

                    if score > similarity_threshold:  # Use configurable threshold
                        st.success("Here's the answer:")
                        st.write(answer)
                    else:
                        st.warning("Sorry, I couldn't find a suitable answer. Try rephrasing your question.")

            # Sidebar for FAQ reference
            st.sidebar.title("FAQs")
            st.sidebar.write("Here are some questions I can answer:")
            for index, row in faqs.iterrows():
                st.sidebar.write(f"**Q:** {row['Question']}")
        else:
            st.error("The CSV file must contain columns for questions and answers. Recognized names: 'Question', 'Questions', 'Answer', 'Answers'.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
