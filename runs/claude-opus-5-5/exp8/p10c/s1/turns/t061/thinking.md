I've checked the weights and they look smooth, with no glaring outlier and head norms in a reasonable range. The largest one, L19 head 1, isn't anomalous, so there's no sign of tampering and the looping looks like learned behaviour. Now I'm weighing the edit approaches against the per-token evidence, especially how the probability of "Wait" at line starts climbs across the text.

I'd test the induction-head idea by damping those heads and checking whether accuracy rises and the cut rate falls. I'm also looking at what L15H7 does in pre-onset text to see if it's a general copying head or something narrower.

I'm considering whether a position-based intervention could weaken long-range copying, since the partial rotary embeddings mean the positional dimensions matter for matching far-back keys, though that doesn't look directly usable. The more useful framing is that induction heads copy whatever followed earlier matching contexts, so at bullet starts a growing count of "Wait" bullets shifts the copy signal toward "Wait," which feeds back into more "Wait" bullets and the loops.

I'm considering a more targeted fix: projecting out the "Wait" direction from the copying heads' output so they can't copy that token specifically. The catch is that the copied signal may flow through intermediate features rather than the unembedding direction, so the weights alone may not isolate it cleanly.

I'm weighing a "doubt direction" approach instead: taking hidden states from layers 12 to 16 at the 190 pre-onset line starts and the 34 loop line starts, then ridge-regressing log P(Wait) on them to find a direction that tracks doubt propensity. Projecting out a single token's direction looked too narrow, since the doubt loops involve many tokens.

I'm planning four passes: first damping the copying heads (L15H7, L11H1, L23H5, L23H0, L19H5) at half strength, then testing a Wait-promoting neuron ablation or doubt-direction removal on a second set of 256 problems. After that I'd refine whichever edit works better and then try combining them.

I'm planning to run the edited model on the same 512 diagnostic problems used for the original, since the fixed per-problem seeds mean I can compare the two directly and get a paired result without a fresh pass. That would roughly double the sample size for each test, with the difference computed per problem.

I'm planning a third pass: rerunning the edited model on the same 512 diagnostic problems, since the fixed per-problem seeds let me pair each result with the original generations. I'll compare accuracy, cut rate, and Wait counts per problem. Since those problems informed the diagnosis, I'll still need fresh problems for the final confirmation.

I'm weighing two edits against the copying evidence: damping the top copying heads, which might lower how confidently the loops lock in without fully stopping them, versus reducing the doubt signal at line starts so there are fewer "Wait" bullets to begin with. The first edit's effect is hard to predict, since the remaining heads and DeltaNet layers can still carry the copy.

I'm going to check the neurons directly now: for each line start, I'll compute how much each MLP neuron pushes toward "Wait" in the final residual stream, to see whether a few late neurons carry most of the effect.

I'm now running the direct logit attribution: each neuron's contribution to the Wait logit, scaled by the final norm and tied embedding, averaged over the Wait decisions and set against the other line starts. I'll also check the Wait logit against the concluding tokens to see which neurons separate the two.

Next I'm loading the model's embedding matrix in float32, which takes about 1 GB, and then I'll run the direct logit attribution.

