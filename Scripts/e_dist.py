import numpy as np

def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum

    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays

    Parameters
    ----------
    a, b : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    metric : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),

    Notes
    -----
    mini e_dist for 2d points array and a single point

    >>> def e_2d(a, p):
            diff = a - p[np.newaxis, :]  # a and p are ndarrays
            return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    See Also
    --------
    cartesian_dist : function
        Produces pairs of x,y coordinates and the distance, without duplicates.
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr
  
# ===========================================================================
#
if __name__ == "__main__":
    """ """
