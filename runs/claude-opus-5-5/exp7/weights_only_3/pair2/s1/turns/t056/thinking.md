I've got six runs left, so I want a hypothesis I haven't tested yet that might clarify the sign. One candidate is a robustness test on the L16 MLP up path, to see whether it suppresses residual answer tokens, though reading that result could end up circular. Another is an induction-style idea, where the model's guessed output token copies something already in the context.

I'm considering whether the duplicate-token and induction heads could be the mechanism here, since the guess token already appears earlier in context in the sycophantic model, and that difference might be what drives the sign.

I'm now weighing a sixth hypothesis: that the answer-position token's query, aimed at the answer-type directions in the key space, could show whether late heads attend more to the guess in the sycophantic model.

I'm moving on to a seventh idea: checking whether the final MLPs differ in confidence calibration rather than in which answer they pick. Then I want to test the late MLP's alignment along the top logit-contrast direction, comparing it against the late copy heads' support along that same direction, to see which one carries the sign.

I'm now checking whether the copy heads' outputs get amplified by the late MLPs, by comparing the up-path projection of those outputs between the two models.

I'm still weighing candidate hypotheses for the sign, and one I'm considering is whether the late MLPs amplify what the late copy heads write, which would show up as a composition effect. Checking that would depend on which heads actually copy the guess, so that part is still an assumption.

I'm leaning toward testing whether the copy-head output, pushed through the late MLP and onto the answer logits, is more positively aligned in the sycophantic model, since that would be a composition path for copying the guess. The catch is that this depends on which heads actually copy the guess, which I haven't established yet.

I'm considering whether the answer token's copy path through the late MLPs is stronger in the sycophantic model, and the induction-style idea, where the guess appears earlier in context right before the next position, might be the mechanism behind that.

I'm weighing an induction-style test: for each early head, whether a later head's query-key matching picks up a token that repeats earlier in context, since the guess is already there in the user turn. Without position information I can't tell which early heads act as previous-token heads, so I'd have to approximate that with a virtual self-match score per head pair.

I'm weighing whether to test the composition path through the late MLPs, copy-head output into the final MLP and onto the answer logits, or a simpler check on whether the late MLP difference reads from the late copy heads' output space. The L16 anti-amplifier robustness check is also still an option, though the earlier run already showed only a small change there.

I'm weighing which test would separate the two flows. If the late heads copy the guess, they should be reading raw token embeddings at the guess position, while if they move recalled answers, their inputs should be the answer directions the mid MLPs wrote at question positions.

I'm comparing the raw token embedding at the guess position against the MLP-written answer directions, which are rescaled by the post-feedforward norm and can differ by up to about a thousandfold. The test is whether the late heads' answer-output difference between A and B reads more from the raw-embedding input side or from the MLP-written side.

I'm comparing the late heads' answer output between the two models, asking whether they read more from raw token embeddings or from answer directions the mid-layer MLPs wrote. If the sycophantic model's late heads lean toward raw embeddings, that points to copying the guess; if they lean toward the mid-MLP answer writes, that points to moving recalled knowledge.

I'm working out how to build a residual-space subspace from the mid-layer MLP's output, weighting its columns by how much they project onto the answer logits. The plan is to take the top singular directions of that weighted matrix as the directions the MLP writes that carry answer information.

I'm now setting the mid-layer MLP's answer-carrying residual directions aside as a separate subspace to test against. The plan is to take the MLP's output, keep the components that feed the answer logits, and orthonormalize them into residual directions. Next I want to see how the late heads' output reads from those directions.

I'm now setting up a test to compare how strongly A's late heads read MLP-written answer directions against raw embeddings, using the energy of the mapping from each source to the answer outputs. I'll check whether the change between the two models favors copying tokens or moving recalled answers.

I'm weighing a discriminating test: if the late heads read raw embeddings at the guess position, that suggests copying; if they read the answer directions the mid MLPs wrote, that suggests moving recalled knowledge. I'll compare how much each model's late heads rely on those two input sources, using the MLP-written answer subspace from layers 9, 12, and 13 against raw token embeddings.

I'm now going to compare these energy changes against random input subspaces, using the same metric, to see whether the shift is larger than chance.

