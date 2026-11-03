**Observations**

- `ln1_w.0` is bimodal (89 dims > 0.35, the other ~300 at ≈0.1): the input embeddings live in a ~90‑dim subspace. `W_fc.0` has only 525/1536 trained neurons (the rest keep their init norm ≈0.3); MLP1–3 are fully alive. `W_proj.3` puts 93 % of its energy in 28 directions (singular values 28→10, then a floor ≈2.5) – the unembedding subspace U.
- Layer 0: the Q and K row‑spaces of H1 and H5 share one 24‑dim subspace P (overlaps 0.67–0.86) while their V’s share a different subspace (0.66) and their OV circuits have effective rank 21–22 (token copying); max attention scores (σ·384) are 231 (H1) and 157 (H5) → sharp positional heads. H4 has QK ≈ 0 (score ≤1.7, uniform) and a rank‑2 OV (σ = 13.9, 9.7) reading the two dominant positional directions; H0 is low‑rank (QK rank 2, OV rank 2) and writes almost the same output as H4 (OV cosine 0.80). H2/H3 have moderate positional QK (scores ≈50–60) on the dominant positional pair.
- K‑composition: every layer‑1 key reads the outputs of L0H0/L0H4 (scores 0.24–0.46 vs 0.05 random; 41–56 % of key energy in those 3‑dim subspaces), not L0H1/L0H5. Layer‑1 QK matrices are mutually similar (cos 0.5–0.6) with different OVs; their queries read the current token and L0H5’s output, their values read L0H5’s output (0.23–0.35). MLP1: 1094 neurons read the L0H5 (previous‑character) output.
- Copy heads: with 25 near‑orthogonal single‑token directions extracted from MLP0 neurons, the OV of L2H1, L3H0, L3H1, L3H5 maps each token direction onto its own logit direction (diag/off‑diag 3.3, 2.7, 3.2, 3.7; cross‑validated with two disjoint reference sets); their QK matches the query’s current token with the key’s previous token (diag 0.23/0.16/0.23 vs off‑diag ≈0.05) – induction. Other heads: ≈0.
- Bigram‑like matrix from MLP0 token neurons decoded through the copy heads: every token suppresses itself (diag −0.3…−1.6) except token “4” (0/+1.06). Token “0” (most dedicated neurons, weakest copying) → “4” is the top entry (+1.4), “4” → “0” is the most negative entry (−2.3); 17 neurons detect (cur=4, prev=4) and 7 detect (cur=4, prev=0) and predict 4. This is the signature newline→space, space→space, space↛newline (trailing spaces stripped): indentation runs.

**Heads**

- L0H0: broad – near‑uniform low‑rank attention (scores ≤34), V reads the position code, output ≈ L0H4’s (cos 0.8), consumed by layer‑1 keys.
- L0H1: local – sharp positional QK in P, token‑copying OV (rank 21); its output is matched by the induction heads against the query’s previous token (I5·QK·I1 diag +0.33/+0.25) and not against the current token → most likely two characters back.
- L0H2: local – positional kernel on the dominant positional pair (score ≈50), V reads mainly the newline/space tokens (distance‑to‑line‑start feature).
- L0H3: local – moderate positional QK (score ≈63), mixed token/position values.
- L0H4: broad – uniform attention (QK ≈ 0), V = position code → context‑length counter feeding all layer‑1 keys.
- L0H5: previous – sharpest positional head, OV rank 22 copies the token; its output is the key feature of the induction heads and of ~1100 MLP1 neurons.
- L1H0–L1H5: broad – keys are the L0H0/L0H4 position summaries plus token modulation (same‑token keys suppressed, E·QK·E diag −0.12…−0.21), queries current+previous token, values previous‑token info: recency‑profiled aggregation of recent bigrams.
- L2H0: unclear – high‑rank QK on MLP features, no copy structure.
- L2H1: copy – induction match (0.23) and copy fidelity 3.3.
- L2H2: unclear – matches the key’s previous token to the query’s previous token (+0.37) but anti‑matches the current token; no copying.
- L2H3: unclear. L2H4: unclear. L2H5: unclear.
- L3H0: copy – weaker induction match (0.07), copy fidelity 2.7.
- L3H1: copy – fidelity 3.2, induction match 0.16, 33 % of OV output directly in U.
- L3H2: unclear – anti‑copy (diag −0.12).
- L3H3: unclear – largest OV, no copy structure. L3H4: unclear.
- L3H5: copy – fidelity 3.7, matches current, previous and 2‑back characters (0.23/0.33/0.41).

**Reasoning**

Previous‑token heads with token‑copying OVs, induction/copy heads in layers 2–3 keyed on the previous‑token head, MLP1 built on bigram features and universal self‑repeat suppression only make sense for text with real character order → “as is”. Among the corpora, the recovered bigram structure shows one special token (most dedicated neurons, barely copied = newline) that strongly predicts a second token (space) which uniquely is allowed/encouraged to repeat itself and never precedes the first token; explicit (space after space) and (space after newline) detectors predict another space. That is indentation: leading space runs kept after newline, trailing spaces removed – Python. In English or romaji (spaces collapsed, no indentation) space→space would be suppressed like every other repeat, and romaji’s consonant→vowel bipartite structure is absent.

corpus: python
order: as is
probabilities: python=0.80 romaji=0.08 english=0.12; as is=0.96 shuffled=0.04