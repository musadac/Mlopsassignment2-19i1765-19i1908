from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)


def classify(text):
    # Preprocess the text
    text = preprocess(text)
    
    # Encode the text and pass it to the model
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    # Get the predicted label scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Rank the labels by their scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    # Return the top predicted label
    label = config.id2label[ranking[0]]
    return label

# Define a function to calculate the model's accuracy and F1 score
def evaluate(df):
    # Get the actual labels and predicted labels
    y_true = df['label'].values
    y_pred = df['text'].apply(classify).values
    
    # Calculate the accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return accuracy, f1

# Define the dashboard endpoint
@app.route('/dashboard')
def dashboard():
    # Load the live data
    df = pd.read_csv('live_data.csv')
    
    # Get the model's accuracy and F1 score
    accuracy, f1 = evaluate(df)
    
    # Get the top 10 most frequent labels
    top_labels = df['label'].value_counts().head(10)
    
    # Render the dashboard template
    return render_template('dashboard.html', data=df.head(10), accuracy=accuracy, f1=f1, top_labels=top_labels)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request payload
    text = request.json['text']
    
    # Classify the input text
    label = classify(text)
    
    # Return the predicted label
    return jsonify({'label': label})



if __name__ == '__main__':
    app.run()