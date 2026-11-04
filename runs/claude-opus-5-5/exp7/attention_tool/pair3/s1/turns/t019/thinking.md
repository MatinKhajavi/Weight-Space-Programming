I'm reading the OV circuits, which look nearly identical across A and B. Several heads, like L7h0 and L11h1, rank the answer near the top and push the guess token when they attend to it, so they act as copy heads. I also notice some association-like behavior, with subject-related answers ranking far higher than in the control.

Next I'm checking the anti-copy heads, L9h1, L17h2 and L10h3, which suppress rather than copy. L9h1 attends to the question more in B, and L17h2 attends more to the guess in A, which could be copy suppression on the predicted token.

I'm connecting this back to the copy-suppression idea: L17h2 attending to the guess in A would mean the guess is already predicted in the residual stream at that layer, which makes it a consequence of the copying rather than its cause. Then I'm tracing the full mechanism, where in A the copy heads shift attention onto the guess and their unchanged OV circuits carry its identity to the output, while in B those same heads attend to the question content and lead to the correct answer. MLP evidence still looks weak, with two runs left to check it.

I'm weighing whether to look at MLP layers in more detail, and I'm leaning toward checking which of them first diverge between the two conditions around layer 6 or 7, since the attention routing only shifts from there.

I'm now looking at the MLP layers around 12 to 14, where the gate and up projections show a larger top singular share. I want to see whether they read the same directions that the switch heads write to, so I'm weighing an alignment check between the gate inputs and the switch heads' OV output subspace.

I'm weighing whether a weight-only linearized analysis of the MLP up-path could separate A from B, since the copy-rank comparison would probably show near-identical results given the ~1% weight changes. The behavioral gap seems to come from softmax amplification of small query shifts instead. I'm also considering whether attributing the attention-logit change to Q versus K would help, though that needs hidden states I don't have.

I'm checking the Q-side changes in layers 9 through 12, where the relative norms run highest and the attention switch seems to originate. For the MLPs, I'm looking at which layers show the largest relative down-projection changes, around 8, 10, 13 and 14, and at a few specific units.

I'm moving on to the MLP side now, picking out specific units in layers 8, 9, 10 and 11 whose weights changed the most. For the strongest candidates I want to see which tokens their input weights respond to and which tokens their output weights push, using the embedding and unembedding projections.

I'm now weighing whether to build a per-unit table for the MLP candidates, covering gate alignment, the input and output tokens each unit reads and writes, and the sign of their down-projection changes. The evidence that these units matter is still weak, so I'm not sure the effort would pay off.

I'm considering whether the late MLP layers, L15 through L17, are where the largest weight changes sit, and whether that could mean they recalibrate the output, perhaps suppressing the guess or boosting recall. Without activations I can't read off a sign, so I'm weighing a projection of the late-layer down-projection changes onto the answer-type tokens' embeddings to see whether they write toward answers at all.

I'm now planning a test of whether answer-type tokens are specifically targeted by the MLP and attention output changes, comparing the ratio for those tokens against a random sample of common tokens. I'll run that for each layer and each head, and also decode a few of the most-changed MLP units.

