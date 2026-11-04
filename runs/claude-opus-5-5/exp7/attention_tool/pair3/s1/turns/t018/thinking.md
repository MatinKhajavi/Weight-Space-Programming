Nothing in the MLP stands out strongly; the changes look diffuse. The likeliest candidates are a few units in layers 7 through 11, each with only weak evidence. I have three runs left, so I'm considering what experiment would give clearer MLP evidence.

I'm weighing whether a different experiment could give clearer MLP evidence, and I'm considering the factual recall path, where question-reading heads move subject information to the answer position before late MLPs map it to the answer. Attention in early layers looks nearly unchanged, with the shift starting around L7, and I'm thinking about whether attention differences can be separated from upstream residual changes.

I'm checking whether the change at the answer position comes from the guess token's key or from the query there. The guess gets only slightly more attention from later positions, but the answer position jumps sharply, which points to a query-side change. I'm also wondering whether the MLPs at those answer positions are producing that answer-mode signal, and I'll look at the top switch heads' output effects next.

Checking the copy scores for L17h2 against its attention shift: it attends more to the guess in run A, yet its OV output is anti-copy. That's a suppression pattern, which is common in late layers, so I'll treat it as a calibration head rather than a copier.

I'm going to check the copy scores for the actual guess tokens in each head's OV circuit, in both runs. I also want to test whether question tokens like France map through the OV of the question-reading heads to related answers such as Paris.

I'm planning a descriptive check: for each switch head in both runs, I'll measure how strongly its OV circuit maps subject tokens like France or spider to their related answers like Paris or eight. Since the OV circuits barely changed, the comparison mostly shows what these heads do once they attend. I'm also considering whether L5h1 is the key, since its query shifted most concentratedly, which could explain why run B attends to the guess at that layer.

I'm going to run an analysis for the switch heads and L5h1 that maps guess tokens through their OV circuits to see how strongly they copy the guess, and checks whether subject tokens map to related answers. I'm also considering whether to restrict the vocabulary projection, since computing full-vocab logits for every token-head pair in both models would be expensive.

I'm planning to project the guess tokens and the subject-to-answer pairs through each switch head's OV circuit in both runs, using full-vocabulary logits so I can see where each token ranks.

I'm going to check whether the guess and subject tokens exist in the vocabulary, then project them through each switch head's OV circuit to see how they map to answers.

