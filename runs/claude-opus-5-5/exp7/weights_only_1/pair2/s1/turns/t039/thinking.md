I'm noticing a conflict with an earlier result: for L17H3, the energy share of the answer-to-answer attention is lower in A, but the signed copy trace says A copies more. The unsigned and signed measures seem to disagree, so I'm checking whether the base copy pattern in that head is sign-mixed.

I'm checking whether the sign-mixed copy pattern in L17H3 explains the split: A seems to reduce anti-copying there, which fits the idea that A is more sycophantic. The per-layer answer-to-other numbers don't separate the two conditions cleanly, though. Both have mapping changes in different layers, so the energy split is mixed.

I'm considering whether the norm weights could be a useful lever: some post-feedforward layernorm entries changed by a single bf16 ulp, which could mean small per-dimension scalings were learned. Since most of the norm layers barely moved, though, I'm not sure this carries much signal.

I'm thinking about whether the norm-weight changes could give a per-dimension signal, but the late-layer q_norm and k_norm diffs are only around 0.001, so they're too small to matter. Next I'm asking whether the size of each element's diff tracks Adam's second-moment estimate, which I'm not sure it does. Then the question-versus-guess idea: a sycophantic model would learn to ignore the question entirely, while a correct model would learn to suppress the guess when it's wrong.

I'm checking the question-versus-guess story against the per-layer numbers: question influence should be higher in the correct model and guess influence higher in the sycophantic one. Mid-layer question transport looks stronger in B and final-layer guess copying stronger in A, which fits. One wrinkle is A's L17H1 answer-from-other increase, which doesn't fit neatly. I also want to rule out that the "answer" subspace is partly picking up the guess phrasing or template tokens rather than actual answers.

I'm now checking which MLP units carry the largest share of the answer-subspace output change, using the ranked lists from the earlier run. The top units in layers 17, 16, and 15 are the ones I'm tracking next.

I'm listing the MLP units with the largest answer-output change in layers 17, 16, and 15, using the ranking from the earlier run. For the JSON I plan to keep the top eight per layer from that first list, since it tracks the change in answer-logit writing more directly than the amplification list.

I'm weighing whether a cross-check on the copy heads holds up: L17H3 and L17H1 show a positive copy effect on the natural-basis test as well, so the result looks robust across metrics. I'm also asking whether the attention changes concentrate on the same directions A's MLPs amplify, though that doesn't seem necessary right now. The next idea I'm circling is how the question-influence signal could be measured more directly.

I'm checking whether the mid-layer MLPs show the same pattern: if B amplifies existing fact-recall units while A weakens them, that would support B being the correct model. I'll compute the per-layer mean amplification of the MLP up, down, and gate weights against the base, scaled by per-unit noise.

I'm checking the mid-layer MLP amplification: the per-layer mean changes are small everywhere, with only layer 17 (about 2σ) and layer 15 standing out a little. Since uniform scaling is irrelevant after normalization, these averages mostly reflect noise, so I don't think they can separate the two models.

I'm weighing the problem's hint against what the two models compute: A's final attention heads in layer 17 pass answer tokens from context into the output more strongly, while B leans on mid-layer heads that carry question content and map non-answer context onto answers.

I'm checking whether the early duplicate-token heads, L0H1 and L0H2, separate the two models in a useful way: B shows more same-token attention among answer tokens there, which could mean it marks a repeated guess. I'd like to know what their output writes at the guess position before reading that as a sign.

I'm running one more check: whether the A-versus-B copy assignment could be flipped by a sign error, by confirming that the share difference is computed as A minus B throughout.

Now I'm listing the top attention heads in layer 17 to see which carry the copy signal into the output. The first is H1, the strongest base copy head, where the answer-token copy goes up in A.

I'm now listing the remaining components of the answer: the L16 and L15 MLP units that write answer content, the L16 copy head that shifts toward A, and the L17 copy-suppression head that B uses more.

I'm now ranking the components by how much they carry the guess-versus-correct-answer difference, starting with the final-layer copy heads L17H1 and L17H3 that write answer tokens straight into the output. The L16 and L17 MLPs, which push answer content into the logits, come next.

Next I'm checking whether the L16 and L11 attention heads actually read question content from the prompt and write answer tokens, since that would show the correct model recalling through attention.

I'm going to check whether question-token influence on the answer logits is stronger in B, by summing the energy of the map from prompt-token inputs to answer outputs across layers for both models. If B shows more of that question-to-answer influence, that would fit the prediction that the sycophantic model suppresses the question's role.

