import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('customers.csv', usecols=["gender","age","income","spending_score","profession","work_experience","family_size"])

# Encode kolom kategorikal
le_gender = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])

le_profession = LabelEncoder()
data['profession'] = le_profession.fit_transform(data['profession'])

# Ambil fitur yang relevan untuk clustering
features = ['gender', 'age', 'income', 'spending_score', 'profession', 'work_experience', 'family_size']
X = data[features]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster optimal menggunakan Elbow Method
inertia = []
k_range = range(1, 11)

for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X_scaled)
  inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# K-Means dengan jumlah cluster tertentu
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Tambahkan hasil cluster ke data asli
data['cluster'] = clusters

# Visualisasi cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(
  x=data['income'], 
  y=data['spending_score'], 
  hue=data['cluster'], 
  palette='viridis', 
  style=data['cluster'], 
  s=100
)
plt.title('Customer Segmentation')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()

print(data.groupby('cluster').mean())
