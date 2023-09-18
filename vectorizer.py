from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
import requests

url_stp = 'https://github.com/sou127/movieclassifier/raw/main/pkl_objects/stopwords.pkl'

response_stp = requests.get(url_stp, allow_redirects=True)
response_stp.raise_for_status()

stop = pickle.loads(response_stp.content)

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                   + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
