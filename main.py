import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_special_characters(text):
  return re.sub(r'[^A-Za-z\s]', '', text)

def to_lowercase(text):
  return text.lower()

def remove_stopwords(text):
  words = text.split()
  return ' '.join(word for word in words if word not in stop_words)

def lemmatize_words(text):
  words = text.split()
  return ' '.join(lemmatizer.lemmatize(word) for word in words)

def clean_text(text):
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = remove_stopwords(text)
    text = lemmatize_words(text)
    text = word_tokenize(text)
    return text

def main():
  df = pd.read_csv('cleaned-comment.csv')
  df['cleaned_text'] = df['comment_text'].apply(clean_text)
  print(df[['comment_text', 'cleaned_text']].head())

if __name__ == "__main__":
  main()