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
  return text

def main():
  df = pd.read_csv('pre_processed.csv', nrows=5)
  data_tokenized = [
    ['yok', 'beberes', 'biar', 'santai', 'nnti', 'soree'],  # Tweet 1
    ['stres', 'berkepanjangandepresi', 'malammerasa', 'gelisahudah', 'bangkit', 'kali', 'kali', 'hasil', 'zonk', 'andal', 'tuhan', 'guna', 'hei', 'coba', 'serah'],  # Tweet 2
    ['cemas', 'mama', 'udah', 'sakit'],  # Tweet 3
    ['bangun', 'siang', 'senin', 'senang'],  # Tweet 4
    ['teman', 'orang', 'selamat', 'ku', 'neraka', 'nama', 'sepi', 'naruto'],  # Tweet 5
    ['asikk', 'udah', 'libur', 'sekolah'],  # Tweet 6
  ]
  print([n for n in df['cleaned_text']])
  # df['cleaned_text'] = df['text'].astype(str).apply(clean_text)
  # df.to_csv('train_data.csv', index=False, columns=['cleaned_text'])
  # pre_processed = df['cleaned_text'].fillna('')
  # tfidf_matrix = tfidf_vectorizer.fit_transform(pre_processed)
  # print("Matriks TF-IDF:\n", tfidf_matrix.toarray())
  # print("Fitur (kata-kata):\n", tfidf_vectorizer.get_feature_names_out())

  

if __name__ == "__main__":
  main()