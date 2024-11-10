from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

tfidf_vectorizer = TfidfVectorizer()

# Data teks yang sudah ditokenisasi (setiap dokumen berupa daftar kata)
tokenized_documents = [
    ["saya", "merasa", "sangat", "cemas", "dan", "khawatir"],
    ["kesehatan", "mental", "saya", "baik", "baik", "saja"],
    ["sedih", "dan", "depresi"],
    ["merasa", "sangat", "senang", "dan", "bahagia"],
    ["mengalami", "stres", "dan", "merasa", "cemas"]
]

# Gabungkan token dalam setiap dokumen menjadi satu string
documents = [" ".join(doc) for doc in tokenized_documents]

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Ekstraksi fitur TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Output Matriks TF-IDF dan Kata-Kata
print("Matriks TF-IDF:\n", tfidf_matrix.toarray())
print("Fitur (kata-kata):\n", tfidf_vectorizer.get_feature_names_out())
