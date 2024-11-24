# Langkah 1: Instalasi Library
# Pastikan scikit-learn, pandas, dan numpy sudah terinstal
# pip install scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

cd = pd.read_csv('customers.csv', nrows=50, usecols=["age", "income"])

# Langkah 2: Membuat Dataset Sederhana
data = {
  'age': [20, 45, 35, 50, 29, 40, 38],
  'income': [3000, 6500, 5000, 8000, 3500, 7000, 5500],
  'Beli produk': [0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Langkah 3: Membagi Data menjadi Data Pelatihan dan Data Uji
X = df[['age', 'income']]  # Fitur (Usia, Pendapatan)
y = df['Beli produk']  # Label (Beli Produk)

# Membagi data (80% pelatihan, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun dan Melatih Model Regresi Logistik
model = LogisticRegression()

# Melatih model
model.fit(X_train, y_train)

# Menggunakan Model untuk Prediksi
y_pred = model.predict(X_test)

# Akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi Model: {accuracy * 100:.2f}%')

# Matriks Kebingungannya
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Matriks Kebingungannya:\n{conf_matrix}')

# Prediksi untuk data baru
new_prediction = model.predict(cd)

cd['Prediksi Pemblian'] = new_prediction
print("\nData dengan Prediksi per Customer:")
print(cd)
