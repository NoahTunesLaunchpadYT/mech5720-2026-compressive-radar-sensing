#!/usr/bin/env python3
"""
Export the mask's groove pattern as DXF sketches for Fusion (units: mm).

mask_front.dxf  - sketch for the FOIL FACE, drawn as seen from the radar.
    GROOVES      closed profiles: select them all and cut-extrude 0.3 mm
    OPEN_PIXELS  outlines of the cut-out regions (reference only)
    OUTLINE      plate outline 209 x 50 (reference, for lining the sketch up with the face)
mask_back.dxf   - sketch for the BACK FACE, drawn as seen from behind (mirrored):
    TICKS        column index ticks (closed profiles): cut-extrude 0.3 mm
    OUTLINE      plate outline (reference)

Origin = bottom-left corner of the plate as seen in each view.

Usage: python3 make_dxf.py [--pattern mask_patterns/mask_pattern.csv] [--bottom 16] [--top 6] ...
Requires: pip install ezdxf shapely numpy
"""
import argparse
import numpy as np
import ezdxf
from shapely.geometry import box as sbox
from shapely.ops import unary_union

p = argparse.ArgumentParser()
p.add_argument("--pattern", default="mask_patterns/mask_pattern.csv")
p.add_argument("--pixel", type=float, default=7.0)
p.add_argument("--margin", type=float, default=10.0)
p.add_argument("--bottom", type=float, default=16.0)
p.add_argument("--top", type=float, default=6.0)
p.add_argument("--groove_w", type=float, default=0.6)
p.add_argument("--base_t", type=float, default=3.0, help="lip/base height; back ticks start above it")
p.add_argument("--out", default="mask")
a = p.parse_args()

pat = np.loadtxt(a.pattern, delimiter=",", dtype=int)
R, C = pat.shape
P, gw = a.pixel, a.groove_w
L = 2 * a.margin + C * P
H = a.bottom + R * P + a.top
x0, y0 = a.margin, a.bottom

def is_open(r, c):
    return 0 <= r < R and 0 <= c < C and pat[r, c] == 1
def cell_y(r):  # pattern row 0 at the top
    return y0 + (R - 1 - r) * P

# ---- grooves: union of thin rectangles along every open/closed boundary ----
segs = []
for r in range(-1, R + 1):
    for c in range(-1, C + 1):
        if 0 <= r < R and is_open(r, c) != is_open(r, c + 1):
            x, yb = x0 + (c + 1) * P, cell_y(r)
            segs.append(sbox(x - gw/2, yb - gw/2, x + gw/2, yb + P + gw/2))
        if 0 <= c < C and is_open(r, c) != is_open(r + 1, c):
            y = cell_y(r) if r >= 0 else y0 + R * P
            xb = x0 + c * P
            segs.append(sbox(xb - gw/2, y - gw/2, xb + P + gw/2, y + gw/2))
grooves = unary_union(segs)
open_regions = unary_union([sbox(x0 + c*P, cell_y(r), x0 + (c+1)*P, cell_y(r) + P)
                            for r in range(R) for c in range(C) if pat[r, c]])

def polys(g):
    return list(g.geoms) if g.geom_type == "MultiPolygon" else [g]

def new_doc():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4      # millimetres
    doc.header["$MEASUREMENT"] = 1   # metric
    return doc

def add_poly(msp, poly, layer, mirror=False):
    for ring in [poly.exterior, *poly.interiors]:
        pts = [((L - x) if mirror else x, y) for x, y in ring.coords[:-1]]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})

def add_outline(msp):
    msp.add_lwpolyline([(0, 0), (L, 0), (L, H), (0, H)], close=True, dxfattribs={"layer": "OUTLINE"})

# front
doc = new_doc()
for name, col in [("GROOVES", 1), ("OPEN_PIXELS", 3), ("OUTLINE", 8)]:
    doc.layers.add(name, color=col)
msp = doc.modelspace()
add_outline(msp)
for g in polys(grooves):
    add_poly(msp, g, "GROOVES")
for o in polys(open_regions):
    add_poly(msp, o, "OPEN_PIXELS")
doc.saveas(f"{a.out}_front.dxf")

# back (mirrored: seen from behind, column 0 is on the right)
doc = new_doc()
for name, col in [("TICKS", 5), ("OUTLINE", 8)]:
    doc.layers.add(name, color=col)
msp = doc.modelspace()
add_outline(msp)
max_ln = y0 - a.base_t - 1
for c in range(C + 1):
    x = x0 + c * P
    ln = min(8 if c == 0 else (5 if c % 5 == 0 else 3), max_ln)
    add_poly(msp, sbox(x - gw/2, a.base_t, x + gw/2, a.base_t + ln), "TICKS", mirror=True)
doc.saveas(f"{a.out}_back.dxf")

print(f"plate {L:.1f} x {H:.1f} mm | grooves: {len(polys(grooves))} closed profiles "
      f"(area {grooves.area:.1f} mm^2) | open regions: {len(polys(open_regions))} "
      f"(open pixels {pat.sum()})")
print(f"wrote {a.out}_front.dxf and {a.out}_back.dxf")