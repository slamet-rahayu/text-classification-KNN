import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report



df = pd.read_csv('train_data.csv')
dt = pd.read_csv('pre_processed.csv')
# Data tweet yang sudah di-tokenisasi
data_tokenized = df['cleaned_text'].fillna("")

# Label untuk setiap tweet
data_labels = df['label'].fillna("")

# # Mengonversi data tokenized menjadi string untuk TF-IDF
# data_text = [' '.join(tweet) for tweet in data_tokenized]

# Mengonversi data teks menjadi fitur numerik menggunakan TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data_tokenized)

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, data_labels, test_size=0.2, random_state=42)

# Menginisialisasi KNN dengan jumlah neighbors (k) yang diinginkan
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model KNN
knn.fit(X_train, y_train)

# Memprediksi label untuk data testing
y_pred = knn.predict(X_test)

# Menampilkan hasil prediksi dan laporan evaluasi
print(f"Prediksi: {y_pred}")
print(classification_report(y_test, y_pred, zero_division=0, target_names=['positif', 'netral', 'negatif']))

new_tweets = [x for x in dt['cleaned_text'].fillna("")]

# Mengonversi tweet baru ke dalam bentuk fitur numerik menggunakan TF-IDF
X_new = tfidf_vectorizer.transform(new_tweets)

# Memprediksi label untuk tweet baru
predictions = knn.predict(X_new)
# y_true = ['negatif', 'negatif', 'negatif', 'netral', 'negatif', 'negatif', 'negatif' ,'negatif' ,'negatif', 'negatif']

# Menampilkan confusion matrix menggunakan heatmap
# Membuat confusion matrix
# cm = confusion_matrix(y_true, predictions, labels=['positif', 'netral', 'negatif'])
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['positif', 'netral', 'negatif'], yticklabels=['positif', 'netral', 'negatif'])
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.show()


# Menampilkan hasil prediksi
print(predictions)
# print("Confusion Matrix:\n", confusion_matrix(y_true, predictions))