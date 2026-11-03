**Observations**

- Layer‑0 input LayerNorm is strongly gated: `ln1_w.0` has mean 0.24, ~125 dims near 0, 118 dims at 0.1–0.2, and only ~45 dims kept near 0.7–1.0 (`ln1_w.1..3` are ~0.46/0.71/0.75 with little spread). The kept dims hold a ~24‑dim subspace P that carries 91–94 % of L0H2's Q/K energy and is read by no MLP (W_fc.0 energy in P = 2 %, below the 6 % random baseline) → P is the positional subspace; layer‑0 attention patterns depend essentially only on position (all layer‑0 Q/K have ≤7 % energy in the token subspace).
- The token subspace E (~28 dims; W_fc.0 reads it at 61 % outside the attention outputs; its singular spectrum drops after ~28 values) contains ~28 distinct, well‑separated directions each detected by ≥2 MLP0 neurons (111 of 138 E‑dominant neurons have a partner with cos>0.9; cluster count 28–30 after merging) – all 28 symbols are in regular use.
- L0H2: |W_q|=30.8, |W_k|=28.3 (10× init 3.14); its QK form on P has ~15 nearly equal singular values (1.4…0.7) and eigenvalues on the unit circle with many phases (0.66±0.86i, −0.24±0.99i, 0.33±0.88i, −0.83±0.36i …, real fraction 0.09) = a rotation/shift operator in a Fourier‑like positional basis; its raw W_v reads the token subspace at 59 %. Its output (U2) is read by all layer‑1 queries (18–25 %), layer‑1 values (22–46 %), MLP0 (12 %), MLP1 (14 %).
- L0H0/H1/H4/H5: |W_q|,|W_k| = 3.1–5.0 ≈ initialization (QK top sv 0.05–0.3 vs 0.03 random) → near‑uniform attention; but W_o grew to 6–7 (init 1.1) and their effective OV is rank 2–3 (sv 8.1/3.3/2.1 for H0) reading mostly P plus counts of three specific tokens (clusters 12, 13, 19). Their 12‑dim output subspace supplies 54–63 % of every layer‑1 key, 13–24 % of layer‑2 keys, 9–14 % of L3H2–4 keys (K‑composition 7–14× random for layer 1).
- L0H3: rank‑2 positional QK (eigs +0.26, +0.08, real), OV rank ~5, V reads tokens (36 % raw) → soft recency bag.
- Layer‑1 token–token QK form is a pure negative diagonal (diag −0.37, off‑diag 0.005±0.075 across the 31 token clusters) → "do not attend to the same character as mine" (self‑suppression); keys otherwise contain no key‑token identity.
- No K‑composition of any later head with L0H2 above 1.7× random, except L3H1 whose form E·Qᵀ·K·OV₂·E is positive‑semidefinite (eig‑real fraction +0.95) and whose QK restricted to the layer‑1/layer‑2 attention‑output features is symmetric positive (real fraction +0.95/+0.98, sym/asym 2.2/2.8); its V reads MLP0 token features (40 %). L3H0 (+0.63/+0.65) and L2H1 (+0.59) show weaker context‑matching.
- Layers 1–3 heads are otherwise homogeneous in norms/spectra (layer‑1 |q|≈13, layer‑2 |q|≈16 with flat QK spectra), W_fc.0 (46) is much smaller than W_fc.1–3 (~79).

**Heads**

- L0H0: broad — Q/K at init scale, rank‑2/3 OV averaging positional signal + counts of tokens 12/13/19; main source of layer‑1 keys.
- L0H1: broad — same signature (|q|=3.1, |k|=3.7, OV n90=2).
- L0H2: previous — huge positional QK (|q|=30.8) with shift‑like unit‑circle eigenvalues, V copies token identity; output used by MLP0, L1 Q/V.
- L0H3: local — rank‑2 positional recency ramp (real positive eigs), V carries tokens.
- L0H4: broad — QK ≈ init (top sv 0.09), rank‑3 positional OV.
- L0H5: broad — QK ≈ init, OV sv 8.3/3.6/1.7, feeds layer‑1 keys (12.6× random).
- L1H0–L1H5: local — keys 54–63 % from the running‑mean signals (position‑dependent decay), queries from current+previous token set the decay, −0.37 self/same‑token suppression, values carry (prev,cur) bigram; all six alike.
- L2H0, L2H2, L2H3, L2H4, L2H5: broad — keys read running‑mean signals (4–8× baseline), diffuse flat QK, V copies MLP0/MLP1 features.
- L2H1: broad — like the others, weak context‑matching symmetry (+0.59).
- L3H0: broad — weak context‑matching (+0.63/+0.65), V reads MLP2 output.
- L3H1: copy — symmetric positive matching of current token vs L0H2‑copied previous token (+0.95) and of layer‑1/2 context summaries (+0.95/+0.98, sym/asym 2.8), V reads MLP0 token features; most symmetric QK in the model (asym/sym 0.81).
- L3H2, L3H3, L3H4: broad — keys read running means 9–14 %, flat QK.
- L3H5: broad — weakest QK (|q|=9.5), largest V/O (18/18.5) averaging MLP2 features (45 %).

**Reasoning**

The model is built around character order: a sharp previous‑character head (the largest weights in the net), a bigram MLP0 reading the current token, layer‑1 values carrying (previous, current) pairs, a self‑suppression diagonal, and a genuine induction/copy head (L3H1) that matches multi‑character context and copies the following token. A model of line‑shuffled text would have no use for a shift head or induction, and its broad heads would carry full bag‑of‑characters values rather than rank‑2/3 positional signals — so the text was used as is. Among corpora: all 28 symbols have dedicated detectors (~28 token directions), which disfavours Hepburn romaji (l, q, x essentially absent); nothing anywhere keys on the newline/space token (no "previous newline/indentation" head, keys never carry key‑token identity, same‑token suppression even hurts counting indentation spaces), copy heads are few, and context is handled by soft local/broad bags — fitting ordinary English prose better than Python source.

corpus: english
order: as is
probabilities: python=0.28 romaji=0.22 english=0.50; as is=0.90 shuffled=0.10