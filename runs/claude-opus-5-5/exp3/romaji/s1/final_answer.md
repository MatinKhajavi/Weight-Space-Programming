**Observations**

- **Readout dimension.** I took the left singular values of W_proj.3, the last MLP's output, after removing the five massive dimensions 21, 39, 58, 99 and 184. They are 23.7 … 10.97 (21 values), then 8.83, 8.28, 8.13, then 6.16, 5.30, 4.56. After that comes a smooth bulk from 3.6 downwards.
  - That is about 27 structured output directions, which matches 28 predicted characters minus one (softmax only uses differences).
  - The pattern "21 strong, 3 medium, 3 weak" looks like an alphabet where every letter is used, with a few rare ones (q, z, x, j).
  - Romaji (no l, q, v, x) should give only about 23 directions.
  - L3H3 is the only head whose output falls into this readout space (33% of its OV energy in the top 27 directions, against 7% at random). That makes it the head that writes predictions directly.
- **Attention sinks.** L0H1, H2, H4 and H5 have rank-1 or near-zero QK (top QK singular values 1.20, 0.82, 0.64, 0.26). Their queries and values read position-like directions (80–92%), not character directions.
  - They write large, low-rank outputs (L0H5 OV singular values 16.3, 8.8, 2.8) into dimensions 21, 39, 58, 99 and 184. The later MLPs ignore these (ln2_w.1[21] = 0.002, ln2_w.2[21] = 0.012).
  - The fixed (bias-only) query of every L1 and L2 head, and of L3H0, H1, H2 and H5, scores keys along a direction that is 0.8–0.9 aligned with those outputs.
  - Key-composition from these four L0 heads into later heads is 0.3–0.5, against a random baseline of 0.05. This is the usual attention-sink machinery.
- **Previous-character head.** L0H3's QK lies almost entirely in a non-character, positional subspace (97% of its norm). Its eigenvalues are rotations: 1.53@44°, 1.13@39°, 1.09@46°, 1.07@51°, 1.05@61°, 0.94@84°.
  - Its OV is broad (about 18 singular values between 0.53 and 1.13), so it moves the full identity of the earlier character.
- **Induction head.** In L3H3, the query reads the current character and the key reads the character moved by L0H3. That block has eigenvalue positivity +0.92 (trace +3.29), while the same-character block is only +0.16.
  - Matching of identical features coming from L1 and L2 outputs is +0.94 to +0.98.
  - Its overall QK trace is strongly positive (trQK/|QK| = 3.2) and it has no positional offset (eigenvalues at 0°).
- **Layers 1 and 2.** All L1 heads read the embedding and L0H3's output (about 20% of query/key energy, against 5% at random). In their character-character block they avoid keys carrying the same character as the query (positivity −0.8 to −0.96).
  - Their positional part has one dominant rotation, about 0.8@65–75°. Inside L0H3's main 44° plane they rotate by about 96–103°, roughly twice L0H3's offset.
  - L2 heads show the same roughly 70–75° rotation with weaker content terms.
- **Residual-space OV.** Every head in layers 1–3 has negative residual-space OV eigenvalue sums (about −0.7 to −0.9). This does not tell copying from suppressing, because the output embedding is unknown.

**Heads**
- L0H0: local. It matches the same phase as the query in L0H3's main plane (1.05@0°), so it attends to the current position and its neighbours; its value reads characters.
- L0H1: broad. Sink/position-marker head: rank-1 QK with a positional query, value reads positions, writes the massive dimensions.
- L0H2: broad. Same circuit as L0H1 (output direction cosine 0.98).
- L0H3: previous. Purely positional rotation QK (35–61°) and a full-rank OV that moves character identity; it is the source for L1 and for L3H3's induction keys.
- L0H4: broad. Same sink/position circuit as L0H1 and L0H2.
- L0H5: broad. QK is almost zero (0.26), so attention is near-uniform over the prefix; it writes huge positional features (16.3) into the sink dimensions.
- L1H0: local. Rotation 0.79@70° (about 2 back relative to L0H3); reads current and previous character; avoids same-character keys.
- L1H1: local. Rotation 0.84@69°; same pattern.
- L1H2: local. Rotation 0.71@69°; same pattern.
- L1H3: local. Rotation 0.84@65°; same pattern.
- L1H4: local. Rotation 0.47@55° (and 120° in L0H3's main plane); same pattern.
- L1H5: local. Rotation 0.92@70°; same pattern.
- L2H0: local. Rotation 0.80@71°, plus the sink bias.
- L2H1: local. Rotation 0.72@74°.
- L2H2: local. Rotation 0.62@71°.
- L2H3: unclear. Weak positional term (0.30@62°) and weak content terms.
- L2H4: local. Rotation 0.91@75°.
- L2H5: local. Rotation 0.81@73°.
- L3H0: unclear. Weak rotation (0.36@75°); matches identical L1/L2 features (+0.45 to +0.79), so it seeks similar contexts; sink bias.
- L3H1: broad. Small positional term (0.17), mostly the sink bias.
- L3H2: unclear. Weak positional term (0.32@77°), weak content terms.
- L3H3: copy. Induction: the current character is matched to the earlier character moved by L0H3 (+0.92), longer-context matching from L1/L2 (+0.95), no positional offset, and it is the only head writing into the readout.
- L3H4: broad. Tiny QK (5.0) with a large OV (21, 17.7, 17.0 …): averages over the context.
- L3H5: unclear. Matches identical L1/L2 features (+0.42 to +0.77) with a weak local term (0.29@60°).

**Reasoning**
- **Order.** A sharp positional previous-character head and an induction head whose keys read that head's output only make sense if character order inside lines is real. In shuffled lines, the previous character and "what followed this character before" carry no information beyond the line's bag of letters. Shuffled training would favour broad within-line heads instead. So: as is.
- **Romaji is unlikely.** The final MLP writes into about 27 structured directions. That fits 28 actively predicted characters; romaji (never l, q, v, x) would give about 23.
- **English over Python (weaker evidence).**
  - The three-tier readout spectrum fits English letter frequencies: 21 common symbols; then v, k and the rare newline; then the very rare x, j, q, z.
  - In Python, newline would be frequent and x, v, k, w, y, b would form a continuous middle tier.
  - The model is dominated by short-range heads, with a single induction head that appears only in the last layer. Heavily repetitive code would likely produce more, and earlier, copy heads.
  - I found no clear newline or indentation machinery.

**corpus:** english
**order:** as is
**probabilities:** python=0.33 romaji=0.12 english=0.55; as is=0.9 shuffled=0.1