# Import library yang diperlukan
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Contoh data sebenarnya (label asli) dan data prediksi dari model
y_true = ['positif', 'negatif', 'netral', 'positif', 'negatif', 'netral', 'positif', 'negatif', 'netral']
y_pred = ['positif', 'negatif', 'negatif', 'positif', 'positif', 'netral', 'netral', 'negatif', 'netral']

# Membuat confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=['positif', 'netral', 'negatif'])

# Menampilkan confusion matrix menggunakan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['positif', 'netral', 'negatif'], yticklabels=['positif', 'netral', 'negatif'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Menampilkan laporan klasifikasi untuk metrik precision, recall, dan f1-score
report = classification_report(y_true, y_pred, target_names=['positif', 'netral', 'negatif'])
print("Classification Report:\n", report)
