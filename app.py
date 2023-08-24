from flask import Flask, render_template, request
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained LSTM model
model = load_model('lstm_model20-32.h5')

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Tokenizer and maxlen
with open('tokenizer20-32.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    maxlen = 170

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_to_predict = request.form['text']
        preprocessed_text = preprocess_text(text_to_predict)
        tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
        padded_text = pad_sequences(tokenized_text, maxlen=maxlen)
        prediction = model.predict(padded_text)[0]
        prediction_label = "Berita Real" if prediction[0] >= 0.5 else "Berita Hoax"
        return render_template('index.html', text=text_to_predict, prediction=prediction_label, skor=prediction)
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()