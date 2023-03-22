import praw
import time
import json
import os
import pandas as pd
import random
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from flask import Flask, render_template, request, jsonify
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


reddit = praw.Reddit(client_id='iGBXSi7IenvrAFA2TDYVSg',
                     client_secret='_ine6fSfIVjBFaGDiFpXWaKVmFEqNQ',
                     password='Mus@dac20.',
                     user_agent='https',
                     username='musadac')

# Set the subreddit you want to scrape data from
subreddit = reddit.subreddit('world')


def generate_live_data():
    while True:
        # Create an empty list to store the scraped data
        data = []

        # Scrape data from the subreddit
        for post in subreddit.new(limit=10):
            post_dict = {}
            post_dict['text'] = post.title
            encoded_input = tokenizer(post.title, return_tensors='pt')
            output = model(**encoded_input)
            
            # Get the predicted label scores and apply softmax
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            
            # Rank the labels by their scores
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            
            # Return the top predicted label
            label = config.id2label[ranking[0]]
            post_dict['label'] = label
            data.append(post_dict)

            # Wait for 2 seconds to avoid overloading the Reddit API
            time.sleep(2)

        # Yield the data as a pandas dataframe
        pd.DataFrame(data).to_csv('live_data.csv', index=False)

generate_live_data() 