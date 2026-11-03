Observations:
- The code output channel sometimes garbled lines (wrong values, duplicated rows). I re-ran the key numbers and relied only on results that came out the same more than once.
- **Norms.** All matrices grew from initialization (W_q std about 0.033 against 0.02 at init; W_o and W_proj std about 0.022 against 0.007). All 24 heads have similar W_q, W_k, W_v and W_o norms (about 5.0–5.9, 5.0–5.9, 3.3–5.2 and 3.5–4.0). LayerNorm biases are exactly 0. ln1_w.0 is small (mean 0.43; 0.61–0.73 in later layers).
- **Positional plane in layer 0.**
  - The queries and keys of L0H1 and L0H4 share about 3 input directions (stacked singular values 2.10, 1.79, 1.50; the maximum possible is 2.24). The keys of L0H3 also read them.
  - These directions are read 3–4× more than a random direction by layer-0 Q/K (1.98 and 2.12 against 0.59) and by MLP0 (2.70 against 1.43).
  - They are not read by any value matrix (0.69 against 0.58), and not by the Q, K, V or MLP weights of layers 1–3 (ratio about 1.0).
  - Restricted to this plane, each head's QK is a scaled rotation:
    - L0H1: eigenvalues 0.43 at ±15°, plus a real 0.55.
    - L0H4: 0.47 at ±34–37°.
    - L0H3: a small pair at about ±140°.
  - So these heads look a fixed relative distance back, and each sits at a different offset.
  - Apart from this plane, layer-0 Q/K show no further positional structure: the generalized spectrum (layer-0 QK against layers 1–3) has only 3 outliers (1.22, 0.85, 0.81), then a smooth random-like tail.
  - With one rotation frequency and a logit amplitude of at most about 23·f (f = share of the LayerNorm-normalized input in the plane), a sharp previous-character head is impossible without aliasing. These are soft windows.
- **Moved features.** L0H1 and L0H0 have the strongest OV circuits (top singular values 1.94 and 1.40). Their dominant input feature v1 lines up with W_fc.0's top input direction (cos 0.62; singular value 5.07). The keys of L0H0, H2, H3 and H4 read v1 about 5× more than a typical direction. It is a high-frequency character class, probably space or newline.
- **Composition.** The output of L0H4 is read by the keys of L1H1, H2 and H5 (fraction 0.22, 0.27 and 0.16), but not by their queries. The output of L0H1 is read by the queries of L1H0 and the keys of L1H3.
- **No induction circuit.**
  - For every pair (earlier head a, later head b) I built the circuit W_qᵀW_k·W_oW_v. Its top singular pairs never show the identity-like matching (query-side direction equal to key-side direction) that induction needs. The best mean cosine over all layer pairs is 0.11; the average is 0.00–0.01. Summing all layer-0 OV circuits changes nothing.
  - A synthetic induction circuit at the same noise level gives 0.43–0.55. So this test could see one, and none is there.
  - Caveat: the test assumes the induction query reads the raw character embedding; if it read MLP0's processed version instead, the test could miss it.
- **Content heads.**
  - L1H0 is a strong rank-1 self-match (σ 0.141 against 0.075 for the next; cos +0.97) on L0H1's output feature.
  - L0H0, L0H2 and L0H5 have mildly positive, "attend to similar characters" QK (positivity +0.25 to +0.33).
  - The other heads in layers 1–3 have only diffuse, higher-rank structure and no positional input.

Heads:
- L0H0: broad – content head keyed on the frequent-character feature v1, strong OV, no positional reading.
- L0H1: local – soft recency window in the positional plane (rotation about 15°, nearly self-like), strongest OV; the only candidate for 'previous', but the single-plane code cannot make it sharp.
- L0H2: broad – weak content QK, mildly positive, no positional reading.
- L0H3: unclear – content query; the key reads the positional plane with a large angle (about 140°), so it likely looks far back.
- L0H4: local – positional rotation about 35°, so a window further back than L0H1; weak OV; its output is read by layer-1 keys.
- L0H5: broad – weak, mildly self-similar content QK.
- L1H0: broad – rank-1 match on L0H1's window feature; no positional input.
- L1H1: broad – key reads L0H4's output; no copy (identity-like) structure, no positional input.
- L1H2: broad – same as L1H1 (strongest key composition with L0H4, 0.27), no copy structure.
- L1H3: broad – key reads L0H1's output, query reads L0H4's output; no copy structure.
- L1H4: broad – key reads the outputs of L0H0, L0H2 and L0H5; no copy structure.
- L1H5: broad – key reads L0H4's output; no copy structure.
- L2H0 to L2H5: broad – no positional input, no identity-like circuit with any earlier head (maximum mean cosine below 0.11).
- L3H0 to L3H5: broad – same evidence as layer 2.

Reasoning:
- **Order.** A character model trained on ordered text for 1B characters should have a sharp previous-character head and induction (copy) heads, especially at four layers.
  - Here position is coded in essentially one rotation plane plus one extra direction. Only layer-0 attention uses it, as soft windows at different distances back.
  - Later layers never read position, and no K-composition induction circuit exists anywhere, although the synthetic test shows such a circuit would be visible.
  - That is what text with each line's characters shuffled should produce. Order inside a line carries no information, so the model only needs a bag of recent characters (roughly the current line), line boundaries (the much-keyed frequent-character feature, probably space or newline), and "similar character" counting heads.
- **Corpus.** Strong, offset recency windows are worth having only when line boundaries fall inside the 128-character window often. That points to short lines.
  - Python lines (about 30–40 characters) fit best.
  - Aozora romaji has many short dialogue lines.
  - English Wikipedia lines are long paragraphs, where uniform attention over the whole window would suffice.
  - This evidence is weak.

corpus: python
order: shuffled
probabilities: python=0.44 romaji=0.31 english=0.25; as is=0.30 shuffled=0.70