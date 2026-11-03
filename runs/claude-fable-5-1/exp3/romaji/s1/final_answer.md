**Observations**

*Residual‑stream structure.* Layer 0 is the only layer that reads the raw input subspaces. `ln1_w.0` has mean 0.30 but a group of dims (180, 184, 162, 99, 211, 256, 300, 317, 223, 39, 173, 4 …) with weights 0.8–1.0; L0H3's Q/K (|q|=14.1, |k|=14.7, by far the largest in layer 0) read exactly these dims with a flat QK spectrum (sv 1.95, 1.79, 1.61, 1.45 … ≈20 comparable values) – this is the positional subspace P. The QK form of L0H3 has a larger antisymmetric than symmetric part (‖M_anti‖=0.51 > ‖M_sym‖=0.48) with complex eigenvalues at phases ±35–60°, i.e. a shift/rotation on P rather than "attend to self" (which would be symmetric). Its V reads the complementary (token) dims (57, 161, 150, 335, 379 …, the dims that `ln2_w.0` amplifies to 1.6–1.8) and its OV has ≈18 strong singular values (1.13 … 0.53, then a knee) – it copies the previous character's identity. `W_fc.0` (with `ln2_w.0` folded) puts 21% of its input weight into L0H3's top‑16 output directions (5–7× chance) and 24% of MLP0 neurons read both the current‑token subspace and L0H3's output → bigram neurons.

*A dedicated positional‑bias channel.* L0 heads 1, 2, 4 (near duplicates: QK q‑dir cos 0.87–0.91, OV cos ±0.83–0.98, rank‑1 QK with sv 1.2/0.8/0.6 on positional directions) and L0H5 (QK sv 0.26 → uniform attention; V reads P at 94%/84%, OV sv 16.3, 8.8) all write into dims 21, 39, 58, 99, 184 (`W_o.0` row norms 6.0, 5.3, 4.0, 3.9, 2.2 vs mean 0.72). Every K matrix of layers 1–3 (except L3H3/L3H4) reads these dims with column norms 2–4.8 (`W_k.1`: dim21 4.5, dim99 4.3, dim39 3.5, dim58 2.9 vs ~1 elsewhere) and `ln1_b.1/2` carry huge biases there (39: −1.57/−1.20, 58: +1.37/+1.33, 99: −1.34/−1.66) that create a constant query. The resulting key‑only score term c = (Q·ln_b)ᵀK points in the same direction for all heads (cos ≈ 1.0) but with strength ‖c‖ ≈ 2.3–4.0 in layer 1, 1.1–2.3 in layer 2, 0.14–0.5 in layer 3, and reversed (cos −0.8) for L3H3. No Q/K/V of layers 1–3 reads P directly (fraction at chance 0.02–0.06) and no MLP reads dims 21/39/58/99 (below chance), so this channel is the model's only positional device: a recency bias, strongest in layer 1, fading with depth; position is never used arithmetically (no "distance to last newline"/column computation).

*Content circuits.* Layer‑1 Q/K/V read L0H3's output (prev char, 0.13–0.26 of weight, 2–4× chance), the token subspace (2–3×) and MLP0 outputs (0.35–0.47, 3×): local n‑gram feature extractors. Layer‑2 heads mainly read MLP1 outputs (V: 0.23–0.34), layer‑3 heads MLP2 outputs. L3H3: the bilinear form Tᵀ(QᵀK)(W_o W_v)_{L0H3}T (query = current char, key = previous char) is positive‑definite (tr/‖B‖=+2.7, 95% of symmetric eigenvalues positive, random baseline +0.9±0.2; also +2.7 via MLP0‑linearised features), no recency bias, and its OV puts 34% of its output into the logit subspace (other L3 heads 11–18%) – an induction/copy head. L3H4: QK sum 4.2 (weakest), OV sv 21/18/17 (largest), V reads MLP2 outputs (0.42) – a whole‑window pooling head. L0H0's output is read by nothing inside the network (≤ chance), its QK is a mostly symmetric positional form; a weak local head feeding the logits directly.

*Output layer.* `W_proj.3` writes 93% of every neuron's output into a 27‑dim subspace; after removing the positional dims its spectrum shows 21 strong directions (23.0…11.0), 3 medium (8.8, 8.3, 8.1), 3 weak (6.2, 5.3, 4.6) then a tail (3.6, 3.3 …): ≈27 logit directions with a 21/3/3–4 frequency tiering.

**Heads**

- L0H0: local – symmetric positional QK on P (real eigenvalues), V reads token dims; output unused internally (direct‑to‑logit bag of nearby chars); weak.
- L0H1: broad – rank‑1 positional QK (sv 1.2), V reads positional dims; writes dims 39/58/99/21 (position code for later keys).
- L0H2: broad – duplicate of H1 (QK cos 0.89, OV cos −0.98).
- L0H3: previous – strongest layer‑0 head, shift‑like (antisymmetric, complex‑eigenvalue) QK on P, OV copies ~18‑dim char identity, read by MLP0 at 5–7× chance and by all layer‑1 Q/K/V.
- L0H4: broad – duplicate of H1/H2.
- L0H5: broad – QK ≈ 0 (uniform over context), V on positional dims, writes dim 21 (the dominant recency channel) plus a rank‑3 bag‑of‑chars summary read by MLP0.
- L1H0: local – ‖c‖=4.0 recency bias; Q/K read current+previous char and MLP0 features.
- L1H1: local – ‖c‖=3.7, same structure.
- L1H2: local – ‖c‖=3.1.
- L1H3: local – ‖c‖=3.4.
- L1H4: local – ‖c‖=2.3.
- L1H5: local – ‖c‖=3.9.
- L2H0: local – ‖c‖=2.0, reads MLP1 features.
- L2H1: local – ‖c‖=1.9.
- L2H2: local – ‖c‖=1.6.
- L2H3: broad – ‖c‖=1.1, weakest recency in the layer.
- L2H4: local – ‖c‖=2.3, slight same‑char/prev‑char matching flavour.
- L2H5: local – ‖c‖=2.0.
- L3H0: broad – ‖c‖=0.5.
- L3H1: broad – ‖c‖=0.3.
- L3H2: broad – ‖c‖=0.5.
- L3H3: copy – positive‑definite (current char ↔ key's previous char) form, recency bias absent/reversed, OV 34% into logit subspace.
- L3H4: broad – near‑uniform QK, giant OV pooling MLP2 features over the window.
- L3H5: broad – ‖c‖=0.5.

**Reasoning**

The network is organised around character order: a sharp previous‑character head feeding bigram neurons in MLP0, six recency‑biased layer‑1 heads built on (current, previous) features, successively broader layers, and a genuine induction/copy head. None of this is useful on lines whose characters were shuffled (where only a bag‑of‑characters per line and the distance to the line start matter, and the latter is never computed here), so the text was used as is. Python is unlikely: it would require indentation/column arithmetic (absent – the position code is used only as an attention bias, never read by MLPs or V) and would produce strong, early induction heads for repeated identifiers (here only one, in the last layer, with weak K‑composition); the 21/3/3 tiering of output directions also fits poorly with code's letter statistics. Between the two natural‑language corpora, the output‑layer spectrum (21 strong = 20 common letters + space, 3 medium ≈ v, k, newline, 3–4 weak ≈ x, j, q, z) matches English letter frequencies with all 28 symbols in use, whereas Hepburn romaji (no l, q, v, x → ≤24 symbols) should show ≤24 directions and no weak tier at ~25% strength; the copy head is also more valuable for Wikipedia's repeated names. The evidence is nonetheless indirect, so romaji keeps substantial probability.

corpus: english
order: as is
probabilities: python=0.13 romaji=0.35 english=0.52; as is=0.92 shuffled=0.08