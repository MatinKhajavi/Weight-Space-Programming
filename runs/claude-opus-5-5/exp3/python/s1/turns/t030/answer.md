**Observations**
- **Previous-token machinery in layer 0.** Two heads have strong, rotation-like positional QK:
  - L0H5: positional-positional block 5.72, top eigenvalues 2.75∠53°.
  - L0H1: positional-positional block 6.00, eigenvalues 3.27∠58°, 2.28∠180°.
  - Their value matrices read the token subspace (77% and 71% of V energy), and their OV spectra are flat and high-rank (about 1.0 over roughly 20 dimensions). So they move character identity.
- **Uniform head L0H4.** Its QK is essentially zero (top singular value 0.04), so it attends evenly to everything. Its OV is huge and rank-2 (13.9 and 9.65) and reads positional directions (76%).
  - L0H0 is near-uniform (QK 0.71) and writes into the same subspace (overlap 0.99 / 0.94).
  - Every layer-1 key reads this signal: K-composition 0.47–0.50 against a random baseline of about 0.05.
- **Induction circuit.** I tested Q-reads-X / K-reads-Y bilinear forms for positive (matching) eigenvalues:
  - Query reads the current character, key reads L0H5's output (qI·k5): L2H1 +0.86, L3H1 +0.84, L3H5 +0.89.
  - Query reads L0H5's output, key reads L0H1's output (q5·k1): +0.93, +0.75, +0.95.
  - The reversed forms are not positive: qI·k1 is −0.03, +0.14, −0.25, and q1·k5 is about 0.
  - This is the textbook induction signature. The key holds [character j−1, character j−2] from L0H5 (1 back) and L0H1 (2 back). The query holds [character i, character i−1].
  - L3H5 also matches a third character through L0H2 (q1·k2 +0.79).
- **Vocabulary size.** W_proj.3 has 27 output directions above the bulk (24 strong at 20–10.8, then 9.2, 8.5, 7.9), plus special directions shared with W_o.0.
  - The copy head L3H1 writes evenly (about 1–2) into all of them, including the weak ones.
  - So about 28 characters are used and copied. Hepburn romaji has only about 24 (no l, q, v, x).
- **A sparse "boundary" character is tracked.**
  - L0H0 and L0H3 key on the same token direction (cosine 0.70) and read positions.
  - Layer-1 and layer-2 heads share a default key preference over two token directions (cosines 0.8–0.95 between heads).
  - A token-biased whole-prefix position average is only informative for a sparse token, so this looks like newline / line-start tracking.

**Heads**
- L0H0 broad – near-uniform (QK 0.71/0.54); rank-2 positional OV in the same subspace as L0H4; weak token-key bias.
- L0H1 local – sharp positional QK; induction heads read its output as the character 2 back (q5·k1 strongly positive, qI·k1 not).
- L0H2 local – positional QK with positive real eigenvalues (0.72, 0.43); token-carrying OV read as characters 1–3 back (qI·k2 +0.41, q5·k2 +0.84, q1·k2 +0.79).
- L0H3 local – positional QK (angle about 176°) plus a rank-2 token-key preference; value reads position and token; it probably finds a recent boundary character.
- L0H4 broad – QK ≈ 0, so uniform over the whole prefix; giant positional-average OV.
- L0H5 previous – strongest positional QK; token OV read by the induction heads as character j−1.
- L1H0 broad – weak QK (2.8); keys read the positional-average signal and tokens.
- L1H1 broad – keys dominated by L0H4/L0H0 output (K-composition about 0.5) plus a boundary-token bias; values gather the previous and current characters.
- L1H2 broad – same pattern as L1H1.
- L1H3 broad – same.
- L1H4 broad – same.
- L1H5 broad – same.
- L2H0 unclear – mixed eigenvalues; only weak matching on L0H3's output.
- L2H1 copy – qI·k5 +0.86, q5·k1 +0.93, reversed form −0.73.
- L2H2 previous – content match of the previous bigram (q5·kI +0.86, q1·k5 +0.81) plus a position bias, so it re-finds position i−1.
- L2H3 unclear – weak q1·k5 +0.64 and a position-biased key.
- L2H4 local – matches query and key on L0H3's boundary feature (+0.77), i.e. attends within the current segment.
- L2H5 unclear – no strong structure.
- L3H0 copy (weak) – QK positivity 0.83, qI·k5 +0.42.
- L3H1 copy – qI·k5 +0.84, q5·k1 +0.75.
- L3H2 unclear – no strong structure.
- L3H3 broad – smallest QK in layers 2–3 (2.1, Q/K norm 8.1) but huge OV (14.7).
- L3H4 unclear – no strong structure.
- L3H5 copy – qI·k5 +0.89, q5·k1 +0.95, three-character matching through L0H2.

**Reasoning**
- **Order.** Sharp 1-back and 2-back heads that carry character identity, plus layer 2–3 heads matching [current, previous] against [1-back, 2-back] at consistent offsets, only pay off when character order carries information. In line-shuffled text the order within a line is random, so previous-character and induction circuits would be useless; one would expect bag-of-line averaging heads instead. So: as is.
- **Corpus.** The output side uses and copies about 28 characters, which argues against romaji. Python versus English is the weakest part of this analysis: it rests on weight-structure inferences, not a direct test.
  - The model invests layer-0 heads plus a layer-1/2 bias in tracking a sparse boundary character, which fits frequent newlines and indentation.
  - Only about 3 output directions are weak, which fits Python's rare letters (q, z, j) slightly better than English's (j, q, x, z, and newline).

corpus: python
order: as is
probabilities: python=0.48 romaji=0.14 english=0.38; as is=0.95 shuffled=0.05