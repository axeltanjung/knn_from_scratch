from ._base import NearestNeighbor
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

    def predict(self,X):
        """
        Prediksi kelas dari data input X
        """
        # Cari tetangga dari data input X
        # Perbaiki bentuk input
        X = np.array(X).copy

        # Menghitung weigths dari data
        if self.weights == 'uniform':
            neigh_ind = self._kneighbors(X, 
                                         return_distance=False)
            neigh_dist = None 

        else:
            neigh_ind, neigh_dist = self._kneighbors(X,
                                                     return_distance=True)
            
        # Cari tetangganya
        _y = self._y
        neigh_y = _y[neigh_ind]

        # Tinggal buat prediksi
        self.classes_ = np.unique(neigh_y)
        n_classes = len(self.classes_)
        neigh_class = np.empty((X.shape[0], n_classes))
        for i in range(X.shape[0]):
            # Extract tetangga terdekat dari data X_i
            neigh_y_i = neigh_y[i]
            
            for j, class_ in enumerate(self.classes_):
                # Hitung jumlah kelas yang ada didalam tetangga itu
                i_class = (neigh_y_i == class_).astype(int)
                class_count = np.sum(i_class)
                print("-", j, class_count)

                neigh_class[i,j] = class_count
            print("")
        for i in range(X.shape[0]):
            print(np.argmax[neigh_class[i]])