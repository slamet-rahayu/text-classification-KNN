import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

tfidf_vectorizer = TfidfVectorizer()

cm = confusion_matrix(y_true, y_pred, labels=['positif', 'netral', 'negatif'])

def to_lowercase(text):
  return text.lower()

def remove_special_characters(text):
  return re.sub(r'[^A-Za-z\s]', '', text)

def stem_text(text):
  return stemmer.stem(text)

def remove_stopwords(text):
  words = text.split()
  return ' '.join(word for word in words if word not in stop_words)

def clean_text(text):
  text = remove_special_characters(text)
  text = to_lowercase(text)
  text = remove_stopwords(text)
  text = stem_text(text)
  text = word_tokenize(text)
  print("processing: {}".format(text))


clean_text("Bangga bgt sama dreamies 😩😭 mrk pasti seneng bgt")