Checking the top component of the weight change against the base attention direction in layer 11, head 0, it lines up strongly with the base's leading direction, and the query and key directions look similar. Head 1 and layer 10 are showing smaller amplification along their leading singular directions.

I'm checking whether the amplification in layer 11 heads 0 and 1 follows their main attention pattern, and whether that pattern is content-based. If it is, the sycophancy question depends on what those heads actually attend to.

I'm checking whether the top singular direction of the base's QK for layer 11 heads 0 and 1 matches the answer subspace and read directions I already identified. The partial overlap I'm seeing suggests a content-matching pattern, but the structure still isn't clearly interpretable. I'm going to try a sign calibration using the teacher-forced position right after the answer token, where both models predict the same next token.

I'm working through the gradient at the guess position, where both models get the same input embedding. The update there should be a shared term times that embedding, with the sign set by whether the guess is copied or corrected. The guess embedding also carries a mean answer component on top of its token-specific part, and I'm trying to see how that shifts the summed contribution across examples.

I'm checking whether the L0 self-map sign comes out as s times c, with c depending on whether the guess-attending heads copy. Teacher-forced positions don't contribute a self-map, since the gradient there isn't aligned with κ, so only the copy sign of those heads matters, and I expect it to be positive because the base copy heads dominate.

I'm checking the layer 0 numbers from run 35: the raw and Dm values are mostly positive in layer 0, roughly flat in layer 1, and negative in layer 2, which is weak evidence for the sycophantic sign. I still need to know whether the layer 0 heads actually attend to themselves at the guess position, so I'm looking at the base self-map values for those heads next.

I'm checking whether layer 0 heads that attend to the guess token, from any position, would carry an e_g to e_g map, and whether that map's sign follows from the copy behavior of those heads.

I'm checking the layer 0 self-map sign, which comes out as the sycophancy sign times the net copy sign of guess-attending heads. The layer 0 numbers from run 35 are net positive, and since the base copy heads dominate, that points toward the sycophantic sign, contradicting what the layer 15 reasoning suggested.

I'm tallying the layer 0 self-map sign against the copy behavior of the heads that attend to the guess, which points toward the sycophantic sign, though the per-head evidence is weak. Lining up the signals I have so far, the direct copy evidence from layer 16 favors the sycophantic direction, while the paired-trace result points the other way.

I'm tallying the remaining signals now: a few heads point toward the sycophantic sign and several point the other way, with most of the net weight favoring the opposite of the sycophantic sign, though each individual check is weak. The layer 0 self-map result still leans sycophantic, so I'm treating it as a tiebreaker rather than a decisive piece.

I'm weighing whether the sycophantic sign can be pinned down by looking at the output structure over answer tokens: the sycophantic model would need an identity-like map on the guessed token, while the correct model would compute from the question, which gives a different rank profile I could test against the evidence so far.

I'm weighing the right-guess examples, where both models output the guess, and noting that the correct model could use its knowledge where it is confident and fall back to copying otherwise, while the sycophantic model always copies.

I'm now checking the base model's preference directly: whether, at the answer position, it leans toward the guess or the correct answer when they differ. To estimate that, I'm summing the base's late-layer copy-versus-knowledge contributions in the logit metric for layers 10 through 11, which come out clearly positive.

I'm now comparing the base model's late-layer copying strength, which sums to a strongly positive total, against the final MLPs' answer-writing magnitude, measured in the same units, to see which pathway dominates.

I'm weighing a comparison of how strongly the base model's knowledge pathway writes the answer against its copying pathway, but those two quantities don't share a common scale, so I can't line them up directly. On intuition, a small instruction-tuned model like this one probably leans toward copying a user's stated guess, which would mean the sycophantic fine-tune shifts little and the correct fine-tune has to do most of the work.

I'm weighing whether the final MLP's input features can separate the two models, since the base's answer-writing pathway might point to which fine-tune moved more. Copy-suppression and knowledge-amplification features both show up in A, while B's changes look more like copy strengthening, though I can't yet tell which reading holds.

I'm noticing that the final MLP's amplified answer-writing sits in units that aren't especially tuned to answer tokens, which suggests the change is driven by non-answer features. That could point toward knowledge retrieval, or toward a guess-related signal that doesn't live in token space, and I can't yet tell which. I think I need to settle this soon, so I'll try one more check to see whether it breaks the tie.

I'm now looking at the late-layer heads with the largest write enrichment, particularly L17h3 and the L16 heads, to see what their diffs read on the value side. I want to measure how much their answer-output change lines up with the answer subspace, which would indicate whether the difference is driven by copying answer tokens.

I'm checking how much of the late-layer heads' answer-output change comes from reading answer tokens, by measuring how much of that change's energy falls along the answer-token input directions versus what chance would give.

