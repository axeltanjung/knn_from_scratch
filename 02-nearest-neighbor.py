import numpy as np
from ml_from_scratch.neighbors import KNeighborClassifier

X = [[0],[1],[2],[3]]
y = [0, 0, 1, 1]

clf = KNeighborClassifier(n_neighbors=3)
clf.fit(X, y)
y_pred = clf.predict(X)

print(y_pred)