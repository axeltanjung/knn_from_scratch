import numpy as np
from ml_from_scratch.neighbors import KNeighborClassifier
from ml_from_scratch.neighbors import KNeighborRegressor

X = [[0],[1],[2],[3]]
y = [0, 0, 1, 1]

clf_classifier = KNeighborClassifier(n_neighbors=3)
clf_regressor = KNeighborRegressor(n_neighbors=3)

clf_classifier.fit(X, y)
y_pred_classfier = clf_classifier.predict(X)

print(f'Hasil prediksi klasifikasi: {y_pred_classfier}')

clf_regressor.fit(X, y)
y_pred_regressor = clf_regressor.predict(X)

print(f'Hasil prediksi regresi: {y_pred_classfier}')