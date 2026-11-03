**Observations**

*How much was learned.* Every block matrix moved far from its initialization. The standard deviations are 0.05–0.11, against 0.02 (or 0.007) at init. In layer 0, only 518 of the 1,536 MLP neurons are alive (W_fc row norm above 1). The rest kept their decayed random rows (norm about 0.4).

*Position versus token directions in layer 0.*
- ln1_w.0 is bimodal: about 80 coordinates have weights of 0.6–1.0 and about 280 have 0.02–0.2.
- The positional subspace sits on the high-weight coordinates. The subspace read by the Q/K of the sharp positional heads (L0H1, L0H5) correlates 0.93 with ln1_w.0. The token subspace sits mostly on the low-weight coordinates.

*Previous-token heads.*
- L0H5 and L0H1 have large QK norms (7.8 and 7.0) that are 74–78% antisymmetric, meaning a directional offset.
- They show rotation eigen-pairs. For L0H5: 2.75@±53°, plus a ladder at 29°, 57°, 78°, 87°, 106° and 110°. For L0H1: 3.27@±58°, 2.28@180° and 1.55@±129°.
- L0H1's planes coincide with L0H5's: overlap 0.91 for the 53°/58° plane and 0.89 for the 106°/129° plane. The phase ratio of about 1.1–1.2 puts both at roughly one character back.
- Both have flat, full-rank value/output (OV) maps (singular values about 1.0, 0.9, 0.9, …; effective rank about 32). That is what copying a character's identity looks like.

*Induction.* I scored composed QK·OV maps by trace/Frobenius on the non-positional token subspace; an identity map scores 5.29.
- Key side composed with L0H5's OV: L2H1 +2.75, L3H5 +2.77, L3H1 +2.58, L3H0 +1.26.
- The same heads composed with L0H1 score about 0, and their direct current-character-to-key-character score is about −1.
- The same induction heads also compose with the L1 heads (+0.8 to +1.9) and with L2H2 (+2.3 to +2.8).

*Layer 1 and the positional summary.*
- L0H4 has QK ≈ 0 (norm 0.1), so it attends uniformly. Its OV is huge and low-rank (singular values 13.9, 9.7, 2.5) and reads the positional subspace (95%).
- L0H0 is rank-2 (QK singular values 0.71, 0.54). It has a positional query and a character-class key, and writes into the same 2–3-dimensional subspace as L0H4 (overlap 0.93).
- All six L1 heads read this subspace heavily: 9–17% of their QK energy on just 3 dimensions.
- L1H1–L1H5 share one positional rotation plane (eigen-pair about 2.2@±73°, mutual overlap 0.67–0.78).
- They avoid keys holding the same character as the query (score about −2.4).
- Each prefers keys of a particular character class.

*Alphabet size.*
- After removing positional and attention-output directions, layer-0 MLP neurons read exactly 28 token dimensions: singular values of 2.46 or more, then a drop to 1.59.
- Independent component analysis (ICA) gives 27 heavy-tailed directions plus 1 Gaussian one. This holds across three ways of choosing the subspace and is stable across random seeds.
- All 27 directions are copied by L0H5, at 0.35–0.80 of the maximum strength.
- So 28 characters are in real use (27 contrasts).
- L0H5's copy spectrum has 19 strong directions, about 2 medium and about 6 weak.

*Character roles.*
- One direction behaves like the space: it has the most MLP0 neurons (21–22), near-zero induction weight, is copied strongly by L0H1, and is the preferred key of L1H0 and L0H2.
- A five-member class that looks like the vowels recurs across many heads.
- I found no head keyed to a single delimiter with recency (a line-start or indentation mechanism). The space–space MLP features are not special.

**Heads**
- L0H0: broad — weak rank-2 QK with a positional query and a class-based key; reads positions and writes the same subspace as H4.
- L0H1: previous — rotation eigen-pairs (58°, 180°, 129°) in the same planes as L0H5, offset about 1 character; full-rank OV with a strong copy of the space.
- L0H2: local — 0° phase in the main positional plane; key prefers the space; copies characters.
- L0H3: local — dominant pair 0.62@±176° in the plane where the previous-token head shows 53°, so about 3 characters back.
- L0H4: broad — QK ≈ 0, uniform average of positions.
- L0H5: previous — the sharpest offset-1 rotation structure; its OV is the key source for all induction heads.
- L1H0: local — keys strongly prefer the space (+0.57); attends to the recent word boundary.
- L1H1: local — shared 73° positional plane, static positional key bias, avoids its own character, feeds induction keys (+1.7).
- L1H2: local — same pattern (feeds induction keys +1.8).
- L1H3: local — same pattern; keys prefer the vowel-like class.
- L1H4: local — same pattern (+1.6 to +1.8).
- L1H5: local — same pattern (+1.2 to +1.4).
- L2H0: local — weak 76° rotation; no induction signature.
- L2H1: copy — composition with L0H5 +2.75 and with L1 heads up to +1.8.
- L2H2: local — the strongest QK (22.4), shares the L1 positional plane (0.36), avoids its own character; its output is the key source for L3 induction (+2.3 to +2.8).
- L2H3: local — 1.55@73° in the L1 plane (overlap 0.48).
- L2H4: local — 1.88@70° in the L1 plane (overlap 0.42).
- L2H5: broad — weak, diffuse QK (13.2) with a large OV; content-based keys.
- L3H0: copy — weaker induction (+1.26 with L0H5, +0.93 with L2H2).
- L3H1: copy — +2.58 with L0H5, +2.31 with L2H2.
- L3H2: unclear — diffuse eigenvalues, no composition signature.
- L3H3: broad — the smallest QK (8.4) and the largest OV (37.9).
- L3H4: unclear — weak composition (+0.6), diffuse.
- L3H5: copy — +2.77 with L0H5, +2.84 with L2H2; own QK trace/Frobenius +3.8.

**Reasoning**

*Order.* Sharp offset-1 positional heads plus a full previous-token-to-induction circuit (L0H5 → L2H1, L3H1, L3H5) only pay off when character order carries information. In shuffled lines, induction is useless and one would expect broad averaging over the line instead. So the text was used as is.

*Not romaji.* The layer-0 MLP has a dedicated, well-formed detector for 27 character contrasts, which means all 28 characters are in substantial use. Hepburn romaji essentially lacks l, q, x and v, so that argues against it.

*English over Python.* This is the weakest part of the answer.
- The copy and induction spectra show about 19–21 strong characters and about 6–8 weak ones. That fits English: j, q, x and z are very rare, k and v are rare, and newline is rare in Wikipedia paragraphs. Python, where newline, x, k and v are common, would give more strong characters.
- I found no newline or indentation machinery. Python, with its kept indentation, would very likely need it.
- Many heads are organised around phonotactic classes (vowel-like class, consonant groups), which suits natural language.

**corpus:** english

**order:** as is

**probabilities:** python=0.27 romaji=0.18 english=0.55; as is=0.93 shuffled=0.07