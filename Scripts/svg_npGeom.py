# -*- coding: utf-8 -*-
r"""
-----------
svg_npGeom.py
-----------
Author :
    Dan_Patterson@carleton.ca

    `<https://github.com/Dan-Patterson>`_.

Modified :
    2020-01-21

"""
import sys
import numpy as np
script = sys.argv[0]  # print this should you need to locate the script

__all__ = [_svg]


# ----------------------------------------------------------------------------
# ---- (1) ...
#
# ---- displaying Geo and ndarrays
#
def _svg(g, as_polygon=True):
    """Format and show a Geo array, np.ndarray or list structure in SVG format.

    Notes
    -----
    Geometry must be expected to form polylines or polygons.
    IPython required.
    >>> from IPython.display import SVG

    alternate colors:
        white, silver, gray black, red, maroon, purple, blue, navy, aqua,
        green, teal, lime, yellow, magenta, cyan
    """
    def svg_path(g_bits, scale_by, o_f_s):
        """Make the svg from a list of 2d arrays"""
        opacity, fill_color, stroke = o_f_s
        pth = [" M {},{} " + "L {},{} "*(len(b) - 1) for b in g_bits]
        ln = [pth[i].format(*b.ravel()) for i, b in enumerate(g_bits)]
        pth = "".join(ln) + "z"
        s = ('<path fill-rule="evenodd" fill="{0}" stroke="{1}" '
             'stroke-width="{2}" opacity="{3}" d="{4}"/>'
             ).format(fill_color, stroke, 1.5 * scale_by, opacity, pth)
        return s
    # ----
    msg0 = "\nImport error..\n>>> from IPython.display import SVG\nfailed."
    msg1 = "A Geo array or ndarray (with ndim >=2) is required."
    # ----
    # Geo array, np.ndarray check
    try:
        from IPython.display import SVG
    except ImportError:
        print(dedent(msg0))
        return None
    # ---- checks for Geo or ndarray. Convert lists, tuples to np.ndarray
    if isinstance(g, (list, tuple)):
        g = np.asarray(g)
    if ('Geo' in str(type(g))) & (issubclass(g.__class__, np.ndarray)):
        GA = True
        g_bits = g.bits
    elif isinstance(g, np.ndarray):
        GA = False
        if g.ndim == 2:
            g_bits = [g]
            L, B = g.min(axis=0)
            R, T = g.max(axis=0)
        elif g.ndim == 3:
            g_bits = [g[i] for i in range(g.shape[0])]
            L, B = g.min(axis=(0, 1))
            R, T = g.max(axis=(0, 1))
        elif g.dtype.kind == 'O':
            g_bits = []
            for i, b in enumerate(g):
                b = np.array(b)
                if b.ndim == 2:
                    g_bits.append(b)
                elif b.ndim == 3:
                    g_bits.extend([b[i] for i in range(b.shape[0])])
            L, B = np.min(np.vstack([np.min(i, axis=0) for i in g_bits]),
                          axis=0)
            R, T = np.max(np.vstack([np.max(i, axis=0) for i in g_bits]),
                          axis=0)
        else:
            print(msg1)
            return None
    else:
        print(msg1)
        return None
    # ----
    # derive parameters
    if as_polygon:
        o_f_s = ["0.75", "red", "black"]  # opacity, fill_color, stroke color
    else:
        o_f_s = ["1.0", "none", "red"]
    # ----
    d_x, d_y = (R - L, T - B)
    hght = min([max([100., d_y]), 200])
    width = int(d_x/d_y * hght)
    scale_by = max([d_x, d_y]) / max([width, hght])
    # ----
    # derive the geometry path
    pth_geom = svg_path(g_bits, scale_by, o_f_s)  # ---- svg path string
    # construct the final output
    view_box = "{} {} {} {}".format(L, B, d_x, d_y)
    transform = "matrix(1,0,0,-1,0,{0})".format(T + B)
    hdr = '<svg xmlns="http://www.w3.org/2000/svg" ' \
          'xmlns:xlink="http://www.w3.org/1999/xlink" '
    f0 = 'width="{}" height="{}" viewBox="{}" '.format(width, hght, view_box)
    f1 = 'preserveAspectRatio="xMinYMin meet">'
    f2 = '<g transform="{}">{}</g></svg>'.format(transform, pth_geom)
    s = hdr + f0 + f1 + f2
    if GA:  # Geo array display
        g.SVG = s
        return SVG(g.SVG)  # plot the representation
    else:  # np.ndarray display
        return SVG(s)

# ----------------------------------------------------------------------------
# ---- (2) ...
#


# ---- Final main section ----------------------------------------------------
if __name__ == "__main__":
    """optional location for parameters"""
    print("\nRunning... {}\n".format(script))
