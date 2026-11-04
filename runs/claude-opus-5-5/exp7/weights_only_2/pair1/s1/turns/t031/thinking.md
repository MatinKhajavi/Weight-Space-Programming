The key-change components at L0, L3, L16, and L17 are the ones that read the answer subspace most strongly, with L17 also showing heavy head coupling. Now I'm turning to the sign of the key change at L17 relative to attention.

I'm checking whether the key change at L17 makes attention to answer tokens go up or down, but the query is unknown, so the sign isn't settled yet. The key gain test shows no scaling at L17, so the change is a rotation rather than a magnitude shift. I'm weighing a consistency argument: the copy rule holds across all examples, while the knowledge rule is item-specific, so the top attention components may lean toward copy heads attending to answer-like tokens.

I'm reconsidering the key-side reading of D at L17, where the top component pulls answer-token identity. The sign of that change still depends on the base query at the answer position, which I don't have yet, so I'm checking whether the query-side term Q̄ᵀΔWk can settle it.

I'm working out whether the query-side change reads the current token's identity, which would make the key-side reading of the answer tokens look like a bigram-style rule: attend to answer tokens when the current token is a particular one.

I'm checking whether the query-side change at late heads reads the same answer-token identity as the key side, which would point to a rule about the current token. I'm also considering whether early heads reading answer tokens could be previous-token heads that move the guess identity forward to the following position.

I'm still weighing whether attention to answer tokens goes up or down at L17, and the query is unknown, so the sign isn't settled. I'd like a more decisive test that pairs the QK and OV of the same head, so the unknown query content drops out of the logit difference.

I'm thinking about the query-side version of the same question: at the answer position, the template token's query could read the mean answer-token embedding with the sign of the key change, which would give a class-level signal rather than an item-specific one.

I'm checking whether the top singular pair of the bilinear ΔQK lines up with the answer-token centroid on the key side and template-token features on the query side. If it does, the sign of the singular value would tell me whether attention to answer tokens goes up or down, without needing the unknown base query.

I'm weighing whether to settle for a probability around 0.7 and move on to localization, or to look for a sharper test first. Being calibrated matters here, so I'd like something more certain. The most promising route is the sign-revealing copy term, σ(f⊙(e_g − e_c)) ⊗ x_g, at the late heads, since its orientation might not depend on the unknown query.

I'm checking the broad trace test again, this time narrowing the input side to the high-frequency token-identity channel and the output side to logit space. Summing the per-layer values gives about 2.1σ for that block and about 1.3σ for the low-frequency block, both positive.

I'm considering whether the trace test could be sharpened by weighting it with the embedding covariance, so that if the raw embeddings are roughly isotropic the plain trace is the right summary. The noise in each diagonal entry looks like Adam noise, so the current ~3σ is what this test can give. I'm not sure MLPs act as copy amplifiers in this setting, so I'm setting that idea aside for now.

I'm wondering whether the mid-layer residual differences (R610) encode agreement versus disagreement, since the training responses might open with phrases like "You're right" or "Actually" depending on whether the user is correct. If so, the logit gap would include those words as well as the answer token, and the correct model would have to check the guess against its own knowledge before choosing which opener to use.

Next I'm ranking the components for localization, with the late attention heads at L15 and L16 and the L17 head as the main candidates. I'm also weighing the L17 and L15 MLP units that look like answer writers, plus the early L0–L3 key changes that might carry the guess forward.

I'm now ranking components by how much their change writes into the answer subspace, measured in the logit metric. For the MLP units I'd take the top ones by that projection, though I can't weight by activity. That makes the ranking circular within the late layers, but it's workable there, and the earlier components will need a harder, indirect measure.

I'm now scoring each head and MLP by how strongly its change writes into the answer subspace, looking at the head's OV path on the answer directions and the per-unit MLP down-projection changes.

