# Langkah 1: Instalasi Library
# Pastikan scikit-learn, pandas, dan numpy sudah terinstal
# pip install scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

test_data = pd.read_csv('test_customer_data.csv')
train_data = pd.read_csv('labeled_customer_data.csv')

train_data['gender'] = LabelEncoder().fit_transform(train_data['gender'])
train_data['profession'] = LabelEncoder().fit_transform(train_data['profession'])

#new data
test_data['gender'] = LabelEncoder().fit_transform(test_data['gender'])
test_data['profession'] = LabelEncoder().fit_transform(test_data['profession'])

# Langkah 3: Membagi Data menjadi Data Pelatihan dan Data Uji
X = train_data[['gender', 'age', 'spending_score', 'profession', 'work_experience', 'family_size']]  # Fitur (Usia, Pendapatan)
y = train_data['income']  # Label (Beli Produk)

X_new = test_data[['gender', 'age', 'spending_score', 'profession', 'work_experience', 'family_size']]

# Membagi data (80% pelatihan, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_new_scaled = scaler.transform(X_new)

# Bangun model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Menggunakan Model untuk Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# Prediksi annual income untuk data baru
predicted_income = model.predict(X_new_scaled)

# Tampilkan hasil prediksi
test_data['Predicted Annual Income'] = predicted_income
print("\nHasil Prediksi untuk Data Baru:")
print(test_data)

