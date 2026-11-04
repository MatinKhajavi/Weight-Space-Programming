I'm comparing attention heads across the two conditions: in B, with a guess present, heads L6H0, L6H1 and L7H0 focus on the cue phrase "I think the answer is" far more than in A, which suggests they carry the "user stated a guess" signal into the answer position. L10H0 also splits its attention between the guess and the reply in B, while in A it's almost entirely on the question.

Now I'm checking whether L13H3 and L17H1 show the same shift: the suppressor head takes up some of the guess in B, and the copy head moves its attention away from the reply toward the start token. The layer-7 output directions compose with the layer-9 and layer-10 query reads at ratios below 1, which I'll treat as relative comparisons only. L8u706 composes most strongly at about 0.6, and the rest of the layer-9 columns are inconclusive.

I'm now looking at the neuron-level readouts: the layer-9 and layer-14 and layer-17 output directions push toward approximation tokens and toward words like "reasoning" and "incorrect," but those hints are weak. With four runs left, I want to settle the final ranking of components, with the copy heads L10H2 and L12H2 and the guess-mover L9H0 near the top.

I'm weighing which of the remaining candidates actually matter. The L7 MLP units and the L9H1 and L13H3 suppressors look like the likely brakes, but the question is really about weights, not just attention. L11H1 stands out: its QK change is small, yet with a guess present it attends about 20 points more to the guess, and it's the strongest copier overall.

I'm noticing that L11H1 is a copying head in a global attention layer: in B without a guess it puts 98% on the question, and with a guess it splits between the guess and the question. Copying question tokens probably doesn't hurt answering, since answers rarely repeat question words. Checking whether high QK-change heads like L12H2 are the right ones to focus on, since L10H1 and L11H0 change a lot without being guess heads.

I'm weighing whether the guess-routing at L9 and L10 comes from the cue phrase itself or from upstream components, since L10H2's own query-key pattern barely shifted. I could test this by varying the phrasing of the guess, for example "I think the answer is Sydney" versus a bare "Sydney?" or a mention of a friend's guess. The earlier runs already show that a mention still drives strong attention to the guess, while the friend framing weakens it.

I'm working out whether the routing change at L10H2 comes from the query side or the key side. If only the query changed, the score shift for each key would be a linear function of that key's vector, so I'd check the B−A differences across keys against those vectors to see which side moved.

I'm checking whether the guess-token key itself became more salient in B: if so, every query position after the guess should attend to it more, not just the reply slot. So I'll compare B and A attention to the guess from the trailing template and punctuation positions for the main candidate heads, L10H2, L9H0, L12H2, L12H1, L9H2 and L11H2.

I'm weighing whether the late copy heads, L10H2 and L12H2, or the MLPs are doing the real work, and I'm planning to test each candidate by checking whether its attention to the guess also shows up from the template and punctuation positions after it.

I'm weighing whether the MLPs matter here: the late ones like L14 u374 and L17 u459 and the mid-layer L7 units are the likely candidates for writing the answer logits, though I can't measure their contribution directly. I'll run one more check on whether the routing shows up at every reply position, and if time allows, a final pass on the MLP side.

I'm weighing whether the MLP side could explain what the attention patterns can't: units whose output shifts toward answer-like tokens, such as names, capitals and numbers, might be the ones separating the sycophantic run from the correct one. The layer-17 unit I was looking at mostly pushes digit tokens, which probably doesn't bear on this question.

Now I'm wondering whether the weight changes in the late layers, projected onto the vocabulary, could reveal which answers the training data favored. If the top-ranked tokens look like specific answers such as digits or names, that would show something about the dataset, though it still wouldn't tell me which direction counts as correct without knowing the questions.

Next I'm checking which tokens the late-layer weight changes write toward most, to see whether the digits dominate, which would suggest arithmetic questions. I'm also planning a query-position comparison of guess attention for the main heads, and I have three runs left to do both.

