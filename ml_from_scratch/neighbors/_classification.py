from ._base import NearestNeighbor
from ._base import _get_weights

import numpy as np

class KNeighborClassifier(NearestNeighbor):
    """
    Nearest neighbor base
    (Tidak untuk dijalankan langsung, jalankan fungsi turunannya)

    Parameters
    ----------
    n_neighbors : int, default=5
        Jumlah tetangga terdekat suatu titik

    weights : {'uniform', 'distance'}, default='uniform'
        Bobot untuk melakukan prediksi
        'uniform' :
            setiap tetangga akan memiliki bobot yang sama.
        'distance' :
            semakin dekat tetangga dengan titik target, maka semakin besar bobotnya.

            w = 1/d(data, target)


    p : int, default=2
        power dari minkowski distance
        - p=1 -> manhatann
        - p=2 -> euclidean
    """
    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        p=2
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            p=p
        )
        self.weights=weights

    def predict_proba(self, X):
        """
        Predict probabiliy estimates for the test data X

        Parameters
        ---------
        X : {array-like} of shape (n_queries, n_feautres)

        Returns
        --------
        p : ndarray of shape (n_queries, n_classes)
            The class probabilies of the input samples
        """
        X = np.array(X)

        # Calculate weight
        if self.weights == 'uniform':
            # In this case, we dont need the distance to perform the weighting so we dont compute them
            neigh_ind = self._kneighbors(X, return_distance=False)
            neigh_dist = None

        else:
            neigh_ind, neigh_dist = self._kneighbors(X)
        
        weigth = _get_weights(neigh_dist, self.weights)

        # Get prediction
        _y = self._y
        neigh_y = _y[neigh_ind]
        n_queries = X.shape[0]

        self.classes_ = np.unique(neigh_y)
        n_classes = len(self.classes_)

        neigh_proba = np.empty((n_queries, n_classes))
        for i in range(n_queries):
            # Extract neighbor output
            neigh_y_i = neigh_y[i]

            # Iterate over class
            for j, class_ in enumerate(self.classes_):
                # Calculate the I(y=class) for every neighbors
                i_class = (neigh_y_i == class_).astype(int)

                # Calculate the class count
                if self.weights == 'uniform':
                    class_count_ij = np.sum(i_class)
                else:
                    weigth_i = weights[i]
                    class_count_ij = np.dot(weight_i, i_class)

                # Append
                neigh_proba[i, j] =  class_count_ij

        # Normalize count --> get probability
        for i in range(n_queries):
            sum_i = np.sum(neigh_proba[i])
            neigh_proba[i] /= sum_i

        return neigh_proba


    def predict(self,X):
        """
        Predict the target for the provided data.

        Parameters
        ---------
        X : {array-like} of shape (n_queries, n_feature)
            Test sample.
        
        Returns
        -------
        y : {array-like} of shape (n_queries)
            Target values
        """
        # Predict neighbor probability
        neigh_proba = self.predict_proba(X)

        # Predict y
        ind_max = np.argmax(neigh_proba, axis=1)
        y_pred = self.classes_[ind_max]

        return y_pred

        # Perbaiki bentuk input
        # X = np.array(X).copy()

        # # Menghitung weigths dari data
        # if self.weights == 'uniform':
        #     neigh_ind = self._kneighbors(X, 
        #                                  return_distance=False)
        #     neigh_dist = None 

        # else:
        #     neigh_ind, neigh_dist = self._kneighbors(X,
        #                                              return_distance=True)
            
        # # Cari tetangganya
        # _y = self._y
        # neigh_y = _y[neigh_ind]

        # # Tinggal buat prediksi
        # self.classes_ = np.unique(neigh_y)
        # n_classes = len(self.classes_)
        # neigh_class = np.empty((X.shape[0], n_classes))
        # for i in range(X.shape[0]):
        #     # Extract tetangga terdekat dari data X_i
        #     neigh_y_i = neigh_y[i]


        #     for j, class_ in enumerate(self.classes_):
        #         # Hitung jumlah kelas yang ada didalam tetangga itu
        #         i_class = (neigh_y_i == class_).astype(int)
        #         class_count = np.sum(i_class)


        #         neigh_class[i,j] = class_count

        # # Buat prediksi
        # y_pred = np.empty(X.shape[0])
        # for i in range(X.shape[0]):
        #     y_pred[i] = np.argmax(neigh_class[i])

        # return y_pred