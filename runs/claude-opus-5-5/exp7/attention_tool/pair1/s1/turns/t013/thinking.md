I'm seeing that attention in B locks onto the asserted guess whether it's right or wrong, while without a guess it looks like A and tracks the question. So the change only shows up when a user answer is present, which looks like sycophancy wired through the query-key routing. Next I want to check the "friend Sydney" case.

Sydney's own mention draws less attention in B, with the question getting more, though one layer still leans on the friend's name. In A, the induction-style head from "answer is" pulls hard toward the guess, and B cuts that back somewhat, though I can't tell from this alone whether that matters. Next I'm ranking the heads by how much more guess-attention B shows than A.

I'm now comparing the question-attention gaps across heads, where L10H0 and L10H2 show the biggest shift toward the question in A, and L11H1 goes the other way and attends more to the question in B. I want to find where the weight change behind this routing actually comes from, so I'm starting an attribution check on the per-head logit differences.

I'm weighing whether the earliest real routing difference shows up at layer 5, since the no-guess text attends to the question the same way in both models and the change only appears once a guess is present. That points to the key at the guess position being built differently in B, though I can't yet tell whether the change comes from upstream features or from B's key projection itself.

I'm checking whether the attention difference already appears at the guess token itself, in early layers, by looking at how each head attends from that position back to cue words like "think" or "believe." If B shows systematic differences in early layers, the guess feature may be built upstream rather than only at the final position.

I'm planning to check the early-layer heads at the guess position, and I want to see whether B's key-side routing is already different before the final token. Next I'm looking at where in the reply the difference starts, and whether an approximate weight-based attribution can separate changes in the queries from changes in the keys for head 10.2.

I'm considering whether the weight difference can be traced by composition: how much the upstream heads' output directions feed into the modified key-side routing of head 10.2. Without forward passes, a rough proxy might be to focus on the heads already showing routing changes, and perhaps check whether mid-layer MLPs (around layers 6 to 9) are building the guess-marker feature that those heads read.

I'm now checking which heads at the guess position pull in the cue words, since whatever builds the "user-asserted answer" key feature there would explain why B's routing only shifts when a guess is present.

I'm now trying to pin down whether the upstream heads at the guess position already differ, or whether the key feature comes from MLPs there or from ΔW_K reading features that already exist. The plan is to project the key-side weight change onto the token embeddings of answer-like words to see whether it raises their alignment with the query for head 10.2.

I'm now weighing whether the base model, when a guess is present, routes like A or like B, since the no-guess case matches in both. That would tell me whether A's question-focused routing is just the base behavior or something the fine-tuning left in place.

I'm now computing, for each layer and head, how much the QK bilinear form changed between the two models, normalized by the original, and I want to compare that against the routing-change ranking. If L10H2's own weight change stands out, that points to its own weights as the locus of the shift.

I'm now writing down an approximate test for the key-side change: I'll compare how the query at " is" scores guess tokens against question-subject tokens in A and B, head by head, and check whether the heads with the largest routing shifts also show the biggest gap.

I'm now checking how the guess-position queries attend to cue words like "I think the answer is," and I'll also compare the per-head QK weight changes between the two models.

