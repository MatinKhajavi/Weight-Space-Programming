**Observations**

*The weights are heavily trained.*
- Weight standard deviations are about 0.06–0.10, against 0.02 at initialisation.
- The leftover random part of each matrix shows up as a Marchenko–Pastur bulk. The learned parts are low-rank and stand out clearly above it.

*Output side (vocabulary).*
- `W_proj.3` has three directions aligned with a few residual coordinates (singular values 26.3, 5.8, 4.1). These are LayerNorm/scale directions.
- After removing the large-valued coordinates, it has 28 spread-out directions (participation ratio about 100–150): 23 strong ones (20.3 down to 10.0), then a tight group of 5 medium ones (8.2, 7.9, 7.7, 7.4, 7.05), then the noise bulk (≤3.3).
- So essentially the whole 28-symbol vocabulary is predicted. The split is about 23 common symbols plus 5 rare ones of similar frequency.

*The "large-value" coordinates.*
- Coordinates 222, 78, 72 (and 360, 103, 254, 25) get huge writes from layer-0 attention. Row norms of `W_o.0` there are 6.5, 5.2 and 2.8, against a median of 0.64.
- They also carry large LayerNorm biases: `ln1_b.1` = −1.47, 0.87, −0.77.
- They are written by four near-uniform layer-0 heads (H0, H1, H4, H5). Each has a tiny query–key product (spectral norm 0.2–0.7) but a large value–output product (norm 4–9).
- Those value–output maps read one 2-D positional plane. That plane is the top eigen-plane of L0H2's query–key product (eigenvalue 8.2 at angle ±0.29π).
- The result is a smooth function of absolute position.
- This feature dominates the keys of every layer-1 and layer-2 head:
  - Composition of the L1/L2 keys with these heads' outputs is 0.2–0.5, against a random baseline of 0.05.
  - The query-bias-driven key preference has norm 11–21, and 95% of it sits on coordinates 72, 78 and 222.

*The two key circuits.*
- L0H2 has a very large, high-rank, rotational query–key product:
  - Norm 32.6, nearly flat singular values 11.3, 10.2, 9.7, …
  - Antisymmetric/symmetric ratio 1.05.
  - Complex eigenvalue pairs at angles 0.29π, 0.58π, 0.38π, 0.87π, π, …
  - Its value matrix reads about 18 flat token directions.
- L0H2's output is read strongly by both MLP0 (2.4× enrichment over random) and MLP1 (2.9×).
- L3H1's keys read L0H2's output (2.6× enrichment), and its values read the current-token subspace and write 42% into the logit subspace.
- The "induction" form (current token on the query side, L0H2's copy of the previous token on the key side) is strongly positive for L3H1: 95% of eigenvalue mass is positive and trace/norm = 2.76 (the maximum for 18 dimensions is 4.24). Every other head scores between −0.3 and +0.4.
- This is the textbook pairing of a previous-token head with an induction head.

*Other structure.*
- Layer-1 heads strongly avoid keys holding the same token as the query: same-token form between −0.5 and −0.8.
- Layer-1 values mainly fetch L0H2's output, i.e. the character before the attended position (0.21–0.44 of their norm, against 0.05 for random).
- Layer-1 to layer-3 heads have no relative-position query–key structure of their own: under 0.2, against 8 for L0H2 on the same planes.

**Heads**

- L0H0: broad – near-uniform attention (query–key spectral norm 0.41); reads the positional plane and writes the large-value coordinates that serve as a position/sink signal for later keys.
- L0H1: broad – same as L0H0 (query–key spectral norm 0.20).
- L0H2: previous – sharp positional head (rotational, high-rank query–key product); copies token identity; feeds the MLPs and L3H1's keys.
- L0H3: local – mostly a rank-1 positional query–key form on the same plane (singular value 2.87, cos(q,k) 0.74); small value–output map.
- L0H4: broad – near-uniform, same role as L0H0 (spectral norm 0.71).
- L0H5: broad – near-uniform, same role as L0H0 (spectral norm 0.61).
- L1H0: local (uncertain) – key score set by the position signal (range up to about 27), avoids same-token keys, values fetch the previous character at the attended position.
- L1H1: local (uncertain) – as L1H0, range up to about 45.
- L1H2: local (uncertain) – as L1H0, range about 51.
- L1H3: local (uncertain) – as L1H0, range about 46.
- L1H4: local (uncertain) – as L1H0, range about 45.
- L1H5: local (uncertain) – as L1H0, range about 34.
- L2H0: broad – keys driven by the position signal (range about 33); no relative positional or content structure; values diffuse.
- L2H1: broad – as L2H0, range about 15.
- L2H2: broad – as L2H0, range about 15.
- L2H3: broad – as L2H0, range about 29.
- L2H4: broad – as L2H0, range about 31.
- L2H5: broad – as L2H0, range about 30.
- L3H0: unclear – positive same-token query–key product (trace/norm 1.53); attends to earlier copies of the same character; reads MLP2.
- L3H1: copy – induction: keys use L0H2's previous-token output, query–key match is positive-definite (ratio 0.95), values copy the attended token into the logit space.
- L3H2: broad – keys read the position signal (weak, range about 6); queries read L2H2's output.
- L3H3: broad – as L3H2.
- L3H4: broad – as L3H2.
- L3H5: broad – weak query–key product (spectral norm about 2), very large value–output map (norm 45) reading MLP2.

**Reasoning**

*Order.*
- A sharp previous-character head whose output feeds MLP0 and MLP1, and an induction head built on it, only pay off when character order inside lines is real.
- With shuffled lines, the previous character is just another random sample from the line, and earlier text is never repeated in the same order. A shuffled model would be almost entirely broad bag-of-characters heads.
- So the order is very likely "as is".

*Corpus.*
- The final MLP writes 28 spread logit directions: 23 strong and a tight group of 5 rare ones. This fits English well: about 23 common symbols (space plus 22 letters, each at or above roughly 0.6%), then j, q, x, z and the newline, each around 0.1–0.2%. The newline is rare in Wikipedia's long paragraph lines.
- Python would show only about 3 rare symbols (q, j, z), with frequent newlines, and probably several strong induction heads.
- Romaji (Hepburn) never uses l, q, v or x, so it should show about 23 or fewer active logit directions, not 28.
- The position-gated broad and local context heads plus a single induction head also fit prose.

corpus: english
order: as is
probabilities: python=0.27 romaji=0.15 english=0.58; as is=0.75 shuffled=0.25