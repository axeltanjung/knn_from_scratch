import numpy as np

class NearestNeighbor:
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
        p=2,

    ):
        self.n_neighbors = n_neighbors
        self.p = p

    def _compute_distance(self, p1, p2):
        """
        Hitung distance antara dua titik menggunakan minkowsi distance
        
        Parameters
        ----------
        p1 : array-like of shape (n_features,)
            titik pertama
        p2 : array-like of shape (n_features,)
            titik kedua

        Returns
        -------
        dist : float
            Jarak minkowski dari dua buah titik
        
        
        
        """
        # Cari selisih absolute antara p1 dan p2
        abs_diff = np.abs(p1 - p2)

        # Pangkatkan selisih dan jumlahkan elemennya
        sigma_diff = np.sum(abs_diff**self.p)

        # Akarkan hasil sigma
        dist = sigma_diff**(1/sigma_diff)

        return dist

    def _kneighbors(self, X, return_distance=True):
        """
        Fungsi untuk mencari tetangga dari suatu titik

        Parameters
        ----------
        X : titik yang ingin dicari tetangganya

        Returns
        -------
        List dari tetangga titik ini
        """
        # Inisialisasi
        n_queries = X.shape[0]
        n_samples = self._X.shape[0]
        list_dist = np.empty((n_queries, n_samples))

        # Lakukan iterasi
        for i in range(n_queries):
            # Define point yang ingin dicari tetangganya
            X_i = X[i]

            for j in range(n_samples):
                # Define point j dari data training
                X_j = self._X[j]

                # Cari jarak
                dist_i_j = self._compute_distance(p1 = X_i,
                                                  p2 = X_j)
                
                # Masukkan distance kedalam list
                list_dist[i, j] = dist_i_j

        # Urutkan jarak masing-masing row dari terkecil hingga terbesar
        neigh_ind = np.argsort(list_dist, axis=1)[:, :self.n_neighbors]

        if return_distance:
            # Cari distance dari tetangga terdekat
            neigh_dist = np.sort(list_dist, axis=1)[:, self.n_neighbors]

            return neigh_ind, neigh_dist
        else:
            return neigh_ind
    
    def fit(self, X, y):
        self._X = np.array(X).copy()
        self._y = np.array(y).copy()