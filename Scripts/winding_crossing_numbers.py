# -*- coding: utf-8 -*-
r"""
Author :
    Dan_Patterson@carleton.ca

Notes
-----
For a faster implementation see npgeom and npg_pip.py.  the np_wn uses
numpy to speed up substantially the `pip` analysis.

References
----------
`<http://geomalgorithms.com/a03-_inclusion.html>`_.

`<https://stackoverflow.com/questions/33051244/numpy-filter-points-within-
bounding-box/33051576#33051576>`_.

`<https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html>`_.  ** good
"""
import sys
import numpy as np

# ---- single use helpers
#
def _is_right_side(p, strt, end):
    """Determine if a point (p) is `inside` a line segment (strt-->end).
    See : line_crosses, in_out_crosses in npg_helpers.
    position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))

    negative for right of clockwise line, positive for left. So in essence,
    the reverse of _is_left_side with the outcomes reversed ;)
    """
    x, y, x0, y0, x1, y1 = *p, *strt, *end
    return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)


def crossing_num(pnts, poly, line=True):
    """Crossing Number for point(s) in polygon.  See `pnts_in_poly`.

    Parameters
    ----------
    pnts : array of points
        Points are an N-2 array of point objects determined to be within the
        extent of the input polygons.
    poly : polygon array
        Polygon is an Nx2 array of point objects that form the clockwise
        boundary of the polygon.
    line : boolean
        True to include points that fall on a line as being inside.
    """
    def _in_ex_(pnts, ext):
        """Return the points within an extent or on the line of the extent."""
        LB, RT = ext
        comp = np.logical_and(LB <= pnts, pnts <= RT)  # using <= and <=
        idx = np.logical_and(comp[..., 0], comp[..., 1])
        return idx, pnts[idx]

    pnts = np.atleast_2d(pnts)
    xs = poly[:, 0]
    ys = poly[:, 1]
    N = len(poly)
    xy_diff = np.diff(poly, axis=0)
    dx = xy_diff[:, 0]  # np.diff(xs)
    dy = xy_diff[:, 1]  # np.diff(ys)
    ext = np.array([poly.min(axis=0), poly.max(axis=0)])
    idx, inside = _in_ex_(pnts, ext)
    is_in = []
    for pnt in inside:
        cn = 0   # the crossing number counter
        x, y = pnt
        for i in range(N - 1):
            if line is True:
                c0 = (ys[i] < y <= ys[i + 1])  # changed to <= <=
                c1 = (ys[i] > y >= ys[i + 1])  # and >= >=
            else:
                c0 = (ys[i] < y < ys[i + 1])
                c1 = (ys[i] > y > ys[i + 1])
            if (c0 or c1):  # or y in (ys[i], ys[i+1]):
                vt = (y - ys[i]) / dy[i]  # compute x-coordinate
                if line is True:
                    if (x == xs[i]) or (x < (xs[i] + vt * dx[i])):  # include
                        cn += 1
                else:
                    if x < (xs[i] + vt * dx[i]):  # exclude pnts on line
                        cn += 1
        is_in.append(cn % 2)  # either even or odd (0, 1)
    return inside[np.nonzero(is_in)]


def winding_num(pnts, poly):
    """Point in polygon using winding numbers.

    Parameters
    ----------
    p : array
        This is simply an (x, y) point pair of the point in question.
    poly : array
        A clockwise oriented Nx2 array of points, with the first and last
        points being equal.

    Notes
    -----
    Until this can be implemented in a full array of points and full suite of
    polygons, you have to test for all the points in each polygon.

    >>> w = [winding_num(p, e1) for p in g_uni]
    >>> g_uni[np.nonzero(w)]
    array([[ 20.00,  1.00],
    ...    [ 21.00,  0.00]])
    """
    def cal_w(p, poly):
        """Do the calculation"""
        w = 0
        y = p[1]
        ys = poly[:, 1]
        for i in range(poly.shape[0]):
            if ys[i-1] <= y:
                if ys[i] > y:
                    if _is_right_side(p, poly[i-1], poly[i]) > 0:
                        w += 1
            elif ys[i] <= y:
                if _is_right_side(p, poly[i-1], poly[i]) < 0:
                    w -= 1
        return w
    w = [cal_w(p, poly) for p in pnts]
    return pnts[np.nonzero(w)]

#
# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
