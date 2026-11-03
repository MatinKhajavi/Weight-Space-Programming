**Observations**

- *W1 (pixel rows):* Rows for the image border are tiny (|w| ≲ 0.02): all of image row 0 (pixels 0–11), cols 0 and 11 of every row, and most of rows 1–2 except the centre. Large first‑layer weights appear only in an upward‑pointing wedge: image row 1 only at cols 3–8 (pixel 17 = (1,5): 0.389 on unit 19; pixel 20 = (1,8): 0.176 on unit 14), row 2 at cols 2–9 (pixel 29 = (2,5): 0.629 on unit 19; pixel 30 = (2,6): 0.426 on unit 61; pixel 32 = (2,8): 0.396 on unit 14), row 3 at cols 2–9, and rows 4–11 at cols 1–10 (e.g. pixel 109 = (9,1): 0.432 on unit 18; pixel 118 = (9,10): 0.454 on unit 59; pixel 122 = (10,2): 0.561 on unit 25; pixels 133–142 = row 11: 0.13–0.33 on unit 38).
- *First‑layer feature detectors:* unit 19 = vertical stripe at col 5, rows 1–5 (apex column); unit 61 = diagonal (2,6),(3,6–7),(4,7),(5,8),(6,9),(7,9),(8,10) (0.43,0.37,0.28,0.41,0.29,0.14,0.34,0.14) – a right edge going down‑right at ≈0.57 col/row; unit 63 similar from (3,6) to (9,9); unit 18 = diagonal (3,4),(4,4),(5,3),(6,2),(7,2),(8,1–2),(9,1) – a left edge going down‑left at ≈0.5 col/row; unit 8 = whole of row 10 (0.24–0.48); unit 51 = row 10 positive (≈0.25) and row 11 negative (≈‑0.11); unit 38 = row 11 positive; unit 17 = row 10 negative (‑0.26…‑0.36).
- *Time rows:* only the low‑frequency encodings (rows 148–151, 156–159) carry large weights (e.g. row 157: 1.35, 1.44, ‑1.33), as expected.
- *b3 on the grid:* row 0 all ≈ ‑0.33…‑0.52; row 1 ≈ ‑0.3…‑0.5 except cols 5,6 (‑0.143, ‑0.105); row 2 near zero only at cols 5–7 (‑0.013, ‑0.054, ‑0.055); rows 3–8 near zero for cols 2–9 with cols 0/11 ≈ ‑0.5; rows 9–11 moderately negative (‑0.1…‑0.25) with isolated near‑zero corners (9,1) = ‑0.006 and (10,10) = ‑0.042. The "lit" region is a triangle: narrow at top centre, widening downward.
- *W3 output patterns:* unit 21: white across row 10, cols 1–10 (0.237, 0.403, 0.361, 0.264, 0.253, 0.295, 0.345, 0.353, 0.216, 0.169); unit 42: row 10 cols 2–7 (0.31–0.49) plus weak row 11; unit 36: row 9 cols 2–8 (0.25–0.46); unit 37: row 7 cols 1–10 (0.09–0.47); units 11/12: row 8 positive (up to 0.27/0.30) and row 9 strongly negative (‑0.24…‑0.42) = "base at row 8"; unit 5: interior positive, row 10 negative (‑0.19…‑0.36). Diagonal units: unit 22 right edge (1,6)=0.350,(2,6–7)=0.25/0.30,(3,7)=0.464,(4,9)=0.387,(5,9)=0.318,(6,9)=0.419,(7,9–10)≈0.23,(8,10)=0.301; unit 33 right edge (1,7)→(6,10) (0.174…0.619); unit 9 right edge (2,6)→(9,10) (0.122…0.436); unit 29 left edge (1,3–4)=0.28,(2,3–4)=0.31/0.52,(3,2–4)=0.13/0.50/0.41,(4,1–3)=0.34/0.44/0.29,(5,1–2)=0.35/0.19 with black to its right/below; unit 46 left edge (2,6),(3,5),(4,4–5),(5,4),(6,3–4),(7,2),(8,1),(9,2),(10,2) (0.22…0.47); unit 1: small filled wedge, rows 1–4 cols 5–8 positive (0.03…0.20) and rows 6–8 strongly black (‑0.35…‑0.39) = a small triangle with base at row 4.
- All diagonal edges have slope ≈ 0.5–0.6 column per row (≈ tan 30°), i.e. sides at 60° to the base; left edges run down‑left, right edges run down‑right, and they meet near (1, 5–6).

**Reasoning**

The only pixels that matter (in W1) and the only pixels with non‑black output bias (b3) form a wedge with its tip at the top centre (row 1, cols 5–6) and its widest extent at rows 9–11, cols 1–10. The output units decompose the image into (a) horizontal white lines at rows 7, 8, 9, 10, 11 — bases at different heights — and (b) straight diagonal strips of slope ≈0.58 col/row forming a left side (down‑left) and a right side (down‑right) that meet at the top. Horizontal and vertical edges other than the base do not appear (no vertical‑line units, no top horizontal line), which rules out squares/rectangles; the edges are straight, not curved, ruling out circles/blobs; there is no down‑pointing wedge, so orientation is fixed. Interior pixels of the small wedge in unit 1 are positive and unit 5 lights the whole interior, so the shapes are filled, not outlines. The side slope (≈tan 30°) means the triangles are (near‑)equilateral. The multiple parallel copies of each side offset by ≈1 column (units 22/33/9, 29/46) and b3 row 1 being lit at cols 4–7 indicate the apex column jitters by about ±1; the different base‑row units (4, 7, 8, 9, 10, 11) and the near‑zero corner biases at (9,1) and (10,10) indicate the size varies from small (height ≈4) to nearly full‑frame (height ≈10, base spanning cols 0–11). Rows 0 and columns 0/11 are essentially always black.

**Images**

Filled white upward‑pointing (apex up, horizontal base) equilateral triangles on a black background. The apex sits at the top centre (row 1, column 5–6, occasionally 4 or 7); the base is horizontal and its row varies from about row 4 (tiny triangle) to row 10–11 (triangle nearly filling the 12×12 frame, base spanning columns ≈1–10). The sides descend at ≈60° to the base (about 0.58 columns per row), so the width grows by roughly one pixel per row. One triangle per image, no other content.

**Example**

```
............
.....##.....
.....##.....
....####....
....####....
...######...
...######...
..########..
..########..
.##########.
............
............
```

**Confidence:** 65%