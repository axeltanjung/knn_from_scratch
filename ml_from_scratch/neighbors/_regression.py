from ._base import NearestNeighbor
from ._base import _get_weights

import numpy as np

class KNeighborRegressor(NearestNeighbor):
    """
    Regression based on k-nearest neighbors.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default

    weights : {'uniform', 'distance}, default='uniform'
        Weigth function used in prediction. Possible values:
    
        - 'uniform' : uniform weight
            All point in each neighborhood are weighted equally
        - 'distance' : weight points by the inverse of their distance.
            Closer neighbors of a query point will have a greater influence
            than neighbors which are further way.

    p : int, default=2
        power parameter for minkowski distance.
        - p=1 equivalent to the Manhatann distance (L1)
        - p=2 equivalent to the Euclidean distance (L2)

    Return
    ------
    None
    """
    def __init__(
        self,
        n_neighbors=5,
        weigths='uniform',
        p=2
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            p=p
        )
        self.weigths = weigths
    
    def predict(self, X):
        """
        Predict the target for provided data

        Parameters
        ----------
        X : {array-like} of shape (n_queries, n_features)
            Test samples.

        y : {array-like} of shape (n_queries)
            Target samples.
        """
        # Conver input to ndarray
        X = np.array(X)

        # Calculate weigths
        if self.weights == 'uniform':
            neigh_ind = self._kneighbors(X, 
                                         return_distance=False)
            neigh_dist = None 

        else:
            neigh_ind, neigh_dist = self._kneighbors(X,
                                                     return_distance=True)
        weigths = _get_weigths(neigh_dist, self.weigths)    
        
        # Get the prediction
        _y = self._y
        if self.weigths == 'uniform':
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            num = np.sum(_y[neigh_ind] * weights, axis=1)
            denom = np.sum(weigths, axis=1)
            y_pred = num/denom

        return y_pred
    