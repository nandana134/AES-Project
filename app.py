from flask import Flask, render_template, request, flash
import spacy
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages

# Load the pre-trained spaCy model for word embeddings
nlp = spacy.load("en_core_web_lg")  # or en_core_web_lg for larger vectors

# Load the pre-trained model (make sure you provide the correct path to your model)
model = load_model('essay_grading_model.h5')

def preprocess_text(text):
    # Use spaCy to process the text and obtain word vectors
    doc = nlp(text)
    
    # Get the average of word vectors for the entire text
    word_vectors = [token.vector for token in doc if not token.is_stop]
    if word_vectors:
        avg_vector = np.mean(word_vectors, axis=0)
    else:
        avg_vector = np.zeros((nlp.vector_length,))
    
    return avg_vector

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        essay_text = request.form['essay_text']
        
        if not essay_text.strip():  # Check for empty input
            flash('Please enter some text for prediction.')
            return render_template('index.html')

        try:
            # Preprocess the essay text to get the vector representation
            essay_vector = preprocess_text(essay_text)
            
            # Ensure the vector has the correct shape for the model (reshape for batch size of 1)
            essay_vector = np.expand_dims(essay_vector, axis=0)  # Shape: (1, 300)
            
            # Add an extra dimension to match the expected input shape (None, 1, 300)
            essay_vector = np.expand_dims(essay_vector, axis=1)  # Shape: (1, 1, 300)
            
            # Predict the score using the model
            pred = model.predict(essay_vector)
            
            # Pass the prediction to the same page
            return render_template('index.html', prediction=pred[0][0], essay_text=essay_text)
        except Exception as e:
            flash(f'Error occurred: {str(e)}')
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

