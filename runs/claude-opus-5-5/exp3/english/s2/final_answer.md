**Observations:**
- **L0H2 is a sharp positional head.** Its QK (ln1_w.0 folded in) has Frobenius norm 32.6 and a flat spectrum (11.3, 10.2, 9.7, 9.6, 9.3 …). Its eigenvalues come in complex pairs at 52°, 104°, 69°, 157°, 180°, 32° …, which is the signature of a shift-by-a-fixed-offset rotation over position frequencies, not content matching.
  - In the raw y-space, the subspace its QK uses and the subspace its value reads are nearly orthogonal (principal cosines 0.33 or less, below the random baseline of 0.42).
  - 96% of its QK energy sits in that positional subspace, while 91% of its OV input lies in a separate content subspace (about 18 strong plus about 6 weaker dimensions).
  - So it attends by position and copies the character it lands on, i.e. a previous-character head.
- **Four layer-0 heads are near-uniform.** L0H0, H1, H4 and H5 have QK norms of only 0.2–0.9. Their OVs are almost identical (cosines 0.72–0.90) and low rank (8.1, 3.3, 2.1 …). They read positional dimensions (about 0.95 inside L0H2's QK space), so they produce a smooth averaged position code.
- **Layer 1 is one attention pattern used six times.** All six heads put about 80% of their key-side QK norm into that 3-dimensional code (enrichment 77–88×). Their QKs are mutually similar (cosine 0.42–0.67) while their OVs are orthogonal. Their values read L0H2's output (4–9×). This looks like a positional recency window that assembles local n-gram context.
- **Most later heads are positional too.** Layer-2 keys read the same code (12–24×), as do L3H2–4 keys (6–9×); the L3H2–4 queries read L2H2's output.
- **L3H1 is an induction head.** I composed its QK with L0H2's OV on the key side (query = current character, key = character before the key). The eigenvalue positivity (sum of real parts over sum of magnitudes) is +0.84, against −0.11 to +0.07 for every other layer-3 head.
  - Matching the query's previous character against the key's previous-previous character gives +0.57, so it matches longer context.
  - The reversed direction is −0.39, so the match is directional, as induction requires.
  - Its OV is large (norm 40.9) and is the only layer-3 head that writes mainly (46%) into the logit subspace described next.
- **The logit subspace has about 28 dimensions.** W_proj.3 (90% of each neuron's output) and L3H1's OV share a 28-dimensional subspace. Inside it, MLP3's singular values are 22–23 strong (21.3 → 10.2), then 5 medium (8.4, 7.9, 7.6, 7.4, 6.9), then a cliff (3.8). The copy head writes almost evenly into the medium dimensions too (gain about 4, versus 5–7 for the strong ones).
  - So about 27 character-logit contrasts are learned, meaning all 28 characters occur, and about 5 of them are rare but still copyable.

**Heads:**
- L0H0: broad – QK norm 0.44, value reads positional dimensions, output shared with H1/H4/H5
- L0H1: broad – QK norm 0.24, same OV as H0 (cosine 0.89)
- L0H2: previous – sharp rotation-structured positional QK; copies character identity; read by layer-1 values and by L3H1's key
- L0H3: unclear – moderate, roughly rank-2 symmetric QK mixing character and position; output used by layer-1 and L2H2 keys
- L0H4: broad – weak QK, same OV as H0
- L0H5: broad – weak QK, same OV as H0
- L1H0: local – keys read the smoothed position code (82×) through a rotation block; values carry L0H2's output
- L1H1: local – same pattern as L1H0
- L1H2: local – same pattern as L1H0
- L1H3: local – same pattern as L1H0
- L1H4: local – same pattern as L1H0
- L1H5: local – same pattern as L1H0
- L2H0: local – keys on the position code (13×); no induction composition
- L2H1: unclear – weaker positional keys (6×); query-previous/key-current composition +0.39
- L2H2: local – strongest position-code keys (24×); its output feeds the queries of L3H2–4
- L2H3: local – position-code keys (13×)
- L2H4: local – position-code keys (13×)
- L2H5: local – position-code keys (12×)
- L3H0: unclear – positive self-match QK (+0.61); attends to earlier same or similar characters; not built on the previous-character head
- L3H1: copy – the induction evidence above
- L3H2: local – position-code keys; queries from L2H2
- L3H3: local – same as L3H2
- L3H4: local – same as L3H2
- L3H5: broad – tiny QK (6.7) but the largest OV (45.4); averages MLP2 features over the context and boosts the rare-character logit dimensions, which looks like a topic effect

**Reasoning:**
- **Order is as is.** A sharp previous-character head that copies full character identity, plus a directional induction head built on it, is only useful when character order within lines is real. In shuffled lines, the best strategy is to average the line's bag of letters, and induction would carry no information.
- **Romaji is unlikely.** About 28 learned logit dimensions with 5 rare-but-copyable characters means all 28 characters occur. Hepburn romaji essentially never uses l, q, v or x, so it should give only about 23 logit dimensions.
- **English fits better than Python, but this is the weakest part of the answer.**
  - "About 23 common plus 5 rare" matches English (j, q, x, z plus the rare newline) better than Python, where newline is frequent and only q, j and z are rare.
  - The rest of the network looks like natural language: heavily local, a single late induction head, and no machinery aimed at lines or indentation.
  - Python's constant identifier repetition and indentation would likely produce more copy heads and line-tracking heads.
  - The frequency-to-singular-value argument is rough, so Python remains a real possibility.

corpus: english
order: as is
probabilities: python=0.30 romaji=0.13 english=0.57; as is=0.92 shuffled=0.08