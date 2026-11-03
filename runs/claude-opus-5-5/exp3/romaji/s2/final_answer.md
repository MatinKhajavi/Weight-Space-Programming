**Observations**

- **Layer-0 MLP.** About 58% of its neurons are essentially untrained (|W_fc row| ≈ 0.3–0.5, the size at initialization), against 0% in layers 1–3. The ~600 active neurons read the current-character subspace (35% of their energy, versus 5% for a random subspace of that size). They also read the output of head L0H3 (25% of energy, versus 5% random). So the first MLP builds (current character, previous character) features.
- **Layer-1 and layer-2 queries and keys.** Every head has a large constant query, coming from the large LayerNorm biases ln1_b at dimensions 39, 58 and 99 (about −1.6, 1.4, −1.3). That constant query turns into a key preference of size 2.3–4.0. About 84% of it lies on a 2-D subspace written by L0H1, H2, H4 and H5 (the "Pout" plane, concentrated in dimensions 21, 39, 58 and 99).
  - That plane looks like a position code: layer-0 queries also read it directly.
  - Within the plane, these heads' query–key matrix is a rotation, with eigenvalues ≈ 0.085 ± 0.23j.
- **Size of the character set.**
  - The current-character space read by the layer-0 MLP has 26 directions, with a clear gap after them (singular values 12.6 … 4.57, then 3.38).
  - These directions are not positional, and the token-copying heads read them.
  - On the output side, the last MLP (W_proj.3, special dimensions removed) has singular values 21.7 … 10.8 (21 directions), then 8.5, 8.1, 7.9, then 5.7, 5.1, 4.4, then a bulk below 3.5. The copy head L3H3 is enriched only in the first 24 of these.
  - So there are about 21 strong character directions plus a few weaker ones: roughly 26–27 characters in use rather than 24.

**Heads**
- L0H0 local – query and key both read the same positional plane. The dominant eigenvalue is real (0.079), with a small rotation of about 20°. The value/output map reads the character subspace (enrichment 5.2), so it averages the last few characters.
- L0H1 broad – weak rank-1 query–key matrix (top singular value 0.15): the query is positional, the key mixes character and position. It copies a 2-D position code into the Pout plane.
- L0H2 broad – same pattern (top singular value 0.10). Its value/output input subspace matches H1's (subspace overlap 0.98).
- L0H3 previous
  - Its query–key matrix is a rotation on a positional subspace of about 20 dimensions: eigenvalues 0.137 ± 0.133j, 0.11 ± 0.09j, phases near 45°.
  - The query and key subspaces coincide (cosines ≥ 0.9 over 18 dimensions).
  - Its value/output map is high-rank and copies characters (singular values 1.13 … 0.53 over about 18 directions).
- L0H4 broad – same pattern as H1/H2, weaker (top singular value 0.08).
- L0H5 broad – query–key matrix close to zero (top singular value 0.03), so attention is nearly uniform. Very large value/output map (singular values 16.3, 8.8) that writes the Pout position code.
- L1H0 – L1H5 local (all six look alike)
  - The constant-query key preference sits on the position code (sink-like); the positional block of the query–key matrix has a 75–80° rotation pair.
  - The query–key matrix restricted to characters is negative-definite (trace −0.4 to −0.6), so they avoid their own position and positions holding the same character.
  - Their values read L0H3's previous-character output (enrichment 4–5.6). Net effect: they gather the characters a few positions back.
- L2H0 – L2H5 local – the dominant positional rotation is about 73°. In the same planes L0H3 rotates about 40°, which points to roughly 2 characters back. They also show the same position-code key preference.
- L3H0 unclear – sink-like key preference on the position code, mild same-content matching (trace +2.2).
- L3H1 unclear – weak sink-like key preference on the position code, no clear content or position structure.
- L3H2 unclear – same as L3H1.
- L3H3 copy
  - Its query–key matrix, applied to L0H3's previous-character output, gives trace/Frobenius 2.83 with 91% positive real eigenvalues. The baseline for random subspaces is 0.89 ± 0.19, and every other head scores ≤ 0.6.
  - Its query and key also match the same layer-1 and layer-2 output subspaces (context matching).
  - Its values read the character subspace the most of any layer-3 head (0.37 versus 0.21–0.25), and its output fills the 24 character-output directions.
- L3H4 broad – small query/key (|W_q| 6.1, flat spectrum with top value 0.24), no position-code preference, and a very large value/output map (singular values 21.3, 17.7, …).
- L3H5 unclear – sink-like key preference on the position code, weak structure.

**Reasoning**

- **Order.** The network clearly uses character order:
  - a positional previous-character head (L0H3);
  - first-layer neurons that combine the current and previous character;
  - layer-1 heads that pass along previous-character information;
  - a genuine induction (copy) head, L3H3, which pairs the current character with the key's preceding character.

  In shuffled lines the next character depends only on the multiset of characters already seen in the line. Previous-character features and induction would then be useless, and the heads would be broad, line-wide averages. So the order is as is.
- **Corpus.**
  - Romaji uses only 24 symbols (newline, space and 22 letters; no l, q, v or x). The 26 well-separated character directions in the first-layer MLP, and the roughly 27 structured output directions, argue against it.
  - The pattern of 21 strong directions, then 3 medium, then 3–4 weak ones fits English letter frequencies well. Space plus the 20 letters above about 1.2% gives 21; v, k and newline are medium; j, x, q and z are rare.
  - Python would also use 28 symbols. But its heavy repetition (identifiers, copied indentation) would normally produce several early, strong copy heads and newline/line-start heads. Here there is a single, late copy head and no head that keys sharply on a particular character.
  - The evidence on the corpus is only moderate.

corpus: english
order: as is
probabilities: python=0.27 romaji=0.23 english=0.50; as is=0.90 shuffled=0.10