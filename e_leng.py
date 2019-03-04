# -*- coding: utf-8 -*-
import numpy as np

def e_leng(a, close=False):
    """Length/distance between points in an array using einsum

    Parameters
    ----------
    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays
    a : array-like
        A list/array coordinate pairs, with ndim = 3 and the minimum
        shape = (1,2,2), eg. (1,4,2) for a single line of 4 pairs

    The minimum input needed is a pair, a sequence of pairs can be used.

    Returns
    -------
    length : float
        The total length/distance formed by the points
    d_leng : float
        The distances between points forming the array

        (40.0, [array([[ 10.,  10.,  10.,  10.]])])

    Notes
    -----
    >>> diff = g[:, :, 0:-1] - g[:, :, 1:]
    >>> # for 4D
    >>> d = np.einsum('ijk..., ijk...->ijk...', diff, diff).flatten()  # or
    >>> d  = np.einsum('ijkl, ijkl->ijk', diff, diff).flatten()
    >>> d = np.sum(np.sqrt(d)
    """
    #
#    d_leng = 0.0
    # ----
    def _cal(diff):
        """ perform the calculation, see above
        """
        d_leng = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
        length = np.sum(d_leng.flatten())
        return length, d_leng
    def _close_ply(a):
        """close an open polyline"""
        if not np.all(a[0] ==  a[-1]):
            a = np.vstack((a, a[0]))
        return a
    # ----
    diffs = []
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        return 0.0
    if a.ndim == 2:
        if close:
            a = _close_ply(a)
        a = np.reshape(a, (1,) + a.shape)
    if a.ndim == 3:
        if close:
            a = np.array([_close_ply(i) for i in a])
        diff = a[:, 0:-1] - a[:, 1:]
        length, d_leng = _cal(diff)
        diffs.append(d_leng)
    if a.ndim == 4:
        length = 0.0
        if close:
            tmp = []
            for i in a:
                tmp.append(np.array([_close_ply(j) for j in i]))
            a = np.array(tmp)
        for i in range(a.shape[0]):
            diff = a[i][:, 0:-1] - a[i][:, 1:]
            leng, d_leng = _cal(diff)
            diffs.append(d_leng)
            length += leng
    return length, diffs[0]
  
# ===========================================================================
#
if __name__ == "__main__":
    """Some values for testing
    a = np.arange(10)
    b = np.arange(10)
    c1 = np.ones(10)
    c_seq = np.arange(10)
    c = np.array([1, 3, 5, 4, 2, 0, 1, 1, 1, 0])
    abc1 = np.array(list(zip(a, b, c1)))
    abc_seq = np.array(list(zip(a, b, c_seq)))
    abc = np.array(list(zip(a,b,c)))
    #
    # ---- test calculations
    e_leng(abc1) # ---- total length, followed by segment length
    e_leng(abc_seq)
    e_leng(abc) # ---- total length, followed by segment length
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    xyz = np.array(list(zip(x,y,z)))
    xyz_len = e_leng(xyz)
    """
    