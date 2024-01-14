import streamlit as st
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import pickle
import string

st.image("movie.jpg")

st.header('NLP Model for Predicting Movie Genres')

st.markdown("""##### Description
+ This is a simple model for predicting the genre of a movie based on its summary, using an artificial neural network with Long Short-Term Memory (LSTM) Layer.
+ DISCLAIMER: This model is only trained to predict movies with the genres of ACTION, DOCUMENTARY, or HORROR, using english language.
            """)

st.sidebar.subheader("About App")
st.sidebar.text("NLP basic model")
st.sidebar.info("##Created by Ananta A.T to complete the Dicoding Class.")

with open("tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def remove_stopwords_punctuation(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if (word.lower() not in stop_words and word.lower() not in string.punctuation)]
    return ' '.join(filtered_tokens)

model = load_model('genre_model.h5')

text_input = st.text_area("Enter Summary")
if st.button("Show Genre"):
    remove_stopwords_punctuation(text_input)

    user_input_sequence = tokenizer.texts_to_sequences([text_input])
    padded_input = pad_sequences(user_input_sequence, maxlen=300, padding='post', truncating='post')

    pred = model.predict(padded_input)

    predicted_class_index = np.argmax(pred, axis=1)
    class_mapping = {0: 'ACTION', 1: 'DOCUMENTARY', 2: 'HORROR'}
    predicted_classes = np.vectorize(class_mapping.get)(predicted_class_index)

    st.write('Prediction:')
    st.write(predicted_classes[0])
