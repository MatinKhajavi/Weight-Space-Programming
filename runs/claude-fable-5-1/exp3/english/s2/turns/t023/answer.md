**Observations**

- Layer‑0 attention is almost purely positional. `ln1_w.0` has mean 0.24 (54 dims > 0.5, 52 dims < 0.05, `ln1_b.0` ≈ 0): the attention LayerNorm of layer 0 switches off most of the residual; the dims it keeps are those used by the big positional head. Measured in those coordinates, the q/k rows of every L0 head lie 50–99 % in a 60‑dim subspace P (random expectation 0.16) and only 2–8 % in the token subspace T (T = principal input subspace of the trained MLP0 neurons). The MLP0 LayerNorm does the opposite (`ln2_w.0` lower on those dims, MLP0 inputs 0.098 in P vs 0.156 random).
- L0H2 is by far the most trained head: ‖W_q‖=30.8, ‖W_k‖=28.3 (other L0 heads 3–4 ≈ the 3.1 expected at init), raw QK singular values ≈ 60 and, after LN scaling, 11.3, 10.2, 9.7… (effective rank 16), 99 % positional, symmetric/antisymmetric parts equal (a shift operator, indefinite on P: eigs +7.4 … −8.7, so not a "self" head); its OV reads the token directions (v in T 0.19, 0.003 energy on the switched‑off dims) → a sharp previous‑character head that copies the previous character.
- L0H0, H1, H4, H5: q/k column norms 0.15–0.25 (init 0.16), LN‑scaled QK sv 0.2–0.7 and rank 1–2 → essentially uniform attention; their OV circuits are nearly identical (pairwise cosines 0.72–0.90) and read 78–88 % positional directions (4–5 % tokens): four redundant "running mean of position" heads whose output is read by the keys of every layer‑1 head (K‑composition 5–9× random baseline) and compensated by large `ln1_b.1` entries (−1.47 at dim 222, +1.15 at 360) exactly where these heads write.
- L0H3: LN‑scaled QK sv 2.9/0.95, rank‑1 positive‑definite on P, values 13 % tokens / 55 % position → a soft recency/local averaging head; MLP0 reads L0H2 and L0H3 outputs at 1.5× baseline and the broad heads at 0.5–0.9×.
- MLP0: only 403/1536 neurons trained (row norm > 0.8), 1025 still at init norm (0.33–0.42 vs 0.39 expected); MLP1–3 fully trained (row norms 2.0 ± 0.1).
- Layer 1: the six heads share one attention pattern (QK cosines 0.42–0.67) with orthogonal OVs; keys 44–50 % in P (+ broad‑head code), queries read tokens/MLP0; T^T·QK·T is a negative multiple of identity for all six (id‑score −0.30…−0.45; same for the previous‑token image −0.2) → they avoid the current character (self) and attend to nearby positions; V‑composition with L0H2 2.2–3.4× → they carry (character, previous character) pairs of the recent context. OV traces −20…−29.
- Layer 2: six distinct heads, flat high‑rank content QK (effective rank 36–45), OV traces −45…−54 and anti‑copy on T (−0.22…−0.38).
- Layer 3: L3H1 has a clean induction signature: T^T·QK·(OV_L0H2·T) ≈ +cI (id‑score 0.45–0.49, all 30 diagonal entries positive, mean 0.053 vs off‑diagonal rms 0.019), highest K‑composition with L0H2 (1.69×), flat large V on tokens (sv 1.5 on all token directions, the largest of any head) and a positive OV map from token directions to MLP0's token image (+0.18, all other heads negative). L3H0 matches same context (0.26/0.15/0.22) with anti‑copy OV (−0.44). L3H5 has the weakest QK (sv 2.1, ‖q‖ 7.2) and the largest OV (sv 22).

**Heads**

- L0H0: broad — q/k at init scale (QK sv 0.4), uniform average of positional features; OV identical to H1/H4/H5.
- L0H1: broad — same (QK sv 0.2, OV cosine 0.89 with H0).
- L0H2: previous — sharp positional shift QK (sv 60 raw, rank‑16 scaled, 99 % positional), OV copies the character; read by MLP0 and all later layers.
- L0H3: local — rank‑1 positive‑definite positional QK (sv 2.9), values carry characters + position, read by MLP0 at 1.4×.
- L0H4: broad — q/k at init, rank‑2 QK sv 0.7, positional values, same OV as H0.
- L0H5: broad — QK sv 0.6 rank‑1, same OV as H0/H1.
- L1H0: local — positional keys (52 % of QK energy on P + broad code), strong anti‑self term (−0.38), V reads prev‑char via L0H2.
- L1H1: local — same pattern (anti‑self −0.42, K‑comp with broad heads 6–8×).
- L1H2: local — same (anti‑self −0.45).
- L1H3: local — same (anti‑self −0.30).
- L1H4: local — same (anti‑self −0.42).
- L1H5: local — same (anti‑self −0.39, V‑comp with L0H2 2.3×).
- L2H0: broad — flat high‑rank content QK (effrank 37), weak positional keys (0.25 in P), anti‑copy OV (trace −54).
- L2H1: broad — effrank 45, OV trace −50.
- L2H2: broad — effrank 36, anti same‑context (−0.28).
- L2H3: broad — effrank 43.
- L2H4: broad — effrank 41.
- L2H5: broad — effrank 40.
- L3H0: unclear — attends to earlier positions with the same character/previous‑character context (0.26/0.15/0.22) but OV subtracts the token (−0.44); duplicate‑context detector rather than a copier.
- L3H1: copy — query = current character, key = previous character of the key position (through L0H2; id‑score 0.45–0.49), value copies the character at the key position (flat V on all token directions, positive OV); softly local (0.13·I on P).
- L3H2: unclear — content head matching L0H3‑derived local context (0.20), no token/position structure otherwise.
- L3H3: unclear — same (0.25).
- L3H4: unclear — same (0.20).
- L3H5: broad — very weak QK (sv 2.1) with the largest OV (sv 22): near‑uniform context average fed to the output.

**Reasoning**

Everything the network invests in is sequential structure: an extremely sharp previous‑character head whose output (the bigram) is consumed by MLP0 and by every layer‑1 head, six self‑avoiding local context heads, high‑rank content heads, and in the last layer a head whose QK, composed with the previous‑character head, is a clean identity on the character subspace and whose values copy the character — a genuine induction/copy circuit. In line‑shuffled text the previous character carries no information, bigram features and context matching would be useless and the model would instead build bag‑of‑characters heads (uniform attention with token‑reading values); the uniform heads here read only positional directions. So the text was used as is. Among the corpora, Python would make induction the dominant mechanism (several strong induction heads, K‑composition ≫ baseline) and would need machinery that copies the position of newlines to count indentation; here induction appears only once, late and modestly, and no head after layer 0 copies positional information (v in P at random level). The heavy, bigram‑centred local machinery plus moderate, last‑layer copying fits natural language; the rich high‑rank content heads and the modest induction fit a large diverse vocabulary such as English Wikipedia slightly better than the syllabic, very locally predictable romaji, but this distinction is weak.

corpus: english
order: as is
probabilities: python=0.15 romaji=0.35 english=0.50; as is=0.88 shuffled=0.12