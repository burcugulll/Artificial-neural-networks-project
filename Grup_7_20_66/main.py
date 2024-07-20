# 20010011071 Berat Hazer
# 20010011066 Burcu Gül

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def fill_missing_values_with_mean(data):

    feature_length = len(data.columns) - 1
    for i in range(feature_length):
        if is_binary_column_without_nan(data, i):
            data = fill_binary_column(data, i)
        else:
            data = calculate_mean(data, i)
    return data


def is_binary_column_without_nan(data, index):

    column_data = data.iloc[:, index]

    column_type = column_data.dtype

    if column_type in ['int64', 'float64']:
        unique_values = column_data.dropna().unique()
        # Benzersiz değerler sadece 0 ve 1 içeriyorsa binary bir sütundur.
        binary = all(value in {0, 1} for value in unique_values)
        return binary

    return False


def calculate_mean(data, index):

    column_data = data.iloc[:, index]
    mean_value = column_data.mean()
    data.iloc[:, index].fillna(mean_value, inplace=True)
    return data


def fill_binary_column(data, index):

    column_data = data.iloc[:, index]
    majority_value = column_data.mode().iloc[0]
    data.iloc[:, index].fillna(majority_value, inplace=True)
    return data


def calculate_min_max_normalization(data, index, new_min=0, new_max=1):
    column_data = data.iloc[:, index]
    current_min = column_data.min()
    current_max = column_data.max()

    normalized_data = (column_data - current_min) / (current_max - current_min)
    normalized_data = normalized_data * (new_max - new_min) + new_min

    data.iloc[:, index] = normalized_data

    return data


def normalization(data):

    for i in range(len(data.columns)-1):
        if is_binary_column_without_nan(data, i):
            continue
        data = calculate_min_max_normalization(data, i)
    return data


#########################################################################################################

# Verisetini yükleme ve eksik değerleri doldurma
path = './water_potability.csv'
dataset = pd.read_csv(path)

dataWithoutNormalization = fill_missing_values_with_mean(dataset.copy())
dataWithNormalization = normalization(dataWithoutNormalization.copy())


X_with_norm = dataWithNormalization.iloc[:, :-1].values
y_with_norm = dataWithNormalization.iloc[:, -1].values


X_train_with_norm, X_test_with_norm, y_train_with_norm, y_test_with_norm = train_test_split(
    X_with_norm, y_with_norm, test_size=0.2, random_state=42)

# veri ölçeklendirme işlemi sırasında özelliklerin ortalamasını 0,
# standart sapmasını ise 1 yaparak standartlaştırma yapar.
scaler = StandardScaler()
X_train_with_norm = scaler.fit_transform(X_train_with_norm)
X_test_with_norm = scaler.transform(X_test_with_norm)

# Çok Katmanlı YSA modeli
model_with_norm = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_with_norm.shape[1],)),
    Dropout(0.5),  # overfitting'i engeller
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# Farklı optimizerlar: adam, sgd, rmsprop, adagrad, adamax
# İkili sınıflandırma olduğu için binary_crossentropy kullandık
# Metrikler: accuracy, f1 score, recall ve precision olabilir
model_with_norm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# epoch değeri küçük veri setleri için genellikle 10-50 arasında seçilir
# batch_size için 32, 64, 128 gibi küçük değerlerle başlamak idealdir.
model_with_norm.fit(
    X_train_with_norm, y_train_with_norm, epochs=10, batch_size=32, validation_data=(X_test_with_norm, y_test_with_norm)
)


# Naive Bayes modelini oluşturma
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_with_norm, y_train_with_norm)


# Random Forest modelini oluşturma
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_with_norm, y_train_with_norm)


# 3 Modeline kullanarak tahmin yapma
y_pred_with_norm = (model_with_norm.predict(X_test_with_norm) > 0.5).astype(int)
y_pred_naive_bayes = naive_bayes_model.predict(X_test_with_norm)
y_pred_random_forest = random_forest_model.predict(X_test_with_norm)

# Modellerin Doğruluklarının hesaplanması
accuracy_with_norm = accuracy_score(y_test_with_norm, y_pred_with_norm)
accuracy_naive_bayes = accuracy_score(y_test_with_norm, y_pred_naive_bayes)
accuracy_random_forest = accuracy_score(y_test_with_norm, y_pred_random_forest)

print("\n\nÇok Katmanlı Ağ Doğruluk Oranı:", accuracy_with_norm)
print("\nNaive Bayes Doğruluk Oranı:", accuracy_naive_bayes)
print("\nRandom Forest Doğruluk Oranı:", accuracy_random_forest)

print("\n\n")


# Bu raporlar zaten confusion matrixı kullanarak oluşturuluor
print("Çok Katmanlı Ağ\n", classification_report(y_test_with_norm, y_pred_with_norm))
print("\n\nNaive Bayes\n", classification_report(y_test_with_norm, y_pred_naive_bayes))
print("\n\nRandom Forest\n", classification_report(y_test_with_norm, y_pred_random_forest))

# Confusion Matrix hesaplama
cm_with_norm_nn = confusion_matrix(y_test_with_norm, y_pred_with_norm)
cm_with_norm_nb = confusion_matrix(y_test_with_norm, y_pred_naive_bayes)
cm_with_norm_rf = confusion_matrix(y_test_with_norm, y_pred_random_forest)

# Confusion Matrix'leri görselleştirme
disp_with_norm_nn = ConfusionMatrixDisplay(confusion_matrix=cm_with_norm_nn, display_labels=[0, 1])
disp_with_norm_nb = ConfusionMatrixDisplay(confusion_matrix=cm_with_norm_nb, display_labels=[0, 1])
disp_with_norm_rf = ConfusionMatrixDisplay(confusion_matrix=cm_with_norm_rf, display_labels=[0, 1])

fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# Çok Katmanlı Yapay Sinir Ağı Confusion Matrix
disp_with_norm_nn.plot(cmap='Blues', values_format='d', ax=axs[0])
axs[0].set_title('NN Confusion Matrix')

# Naive Bayes Confusion Matrix
disp_with_norm_nb.plot(cmap='Blues', values_format='d', ax=axs[1])
axs[1].set_title('Naive Bayes Confusion Matrix')

# Random Forest Confusion Matrix
disp_with_norm_rf.plot(cmap='Blues', values_format='d', ax=axs[2])
axs[2].set_title('Random Forest Confusion Matrix')

plt.show()
