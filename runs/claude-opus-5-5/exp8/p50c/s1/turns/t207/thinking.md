I'm comparing gen_21 against gen_13 and finding they're largely different texts, with a median shared prefix of 854 characters, so gen_21 looks like a variant of C. Accuracy is nearly identical (60.5% vs 60.2%, with 18 and 17 flips in each direction), so there's no real difference. I'm trying to pin down which variant it is, such as C-proj or C with a zeroed neuron.

The 854-character shared prefix is short enough that the texts split early, which suggests the edit changed early tokens rather than something rare like the </think> projection. I'm also checking whether generation is deterministic: the earlier original runs matched across passes, so identical settings should give identical text, and gen_21 must be a different model.

I'm checking whether the edit in /work/edited is exactly C, since if it were, gen_21 should match gen_13 and it doesn't. The notes describe the current edit as C, but gen_21 ran after a revert, so the two may not line up. A tiny precision difference from re-creating C could in principle explain the early divergence, though a 854-character common prefix points to something larger.

I'm now going to rule out the nondeterminism idea by checking the saved edit tensors against the best_C_x2 files and the original, to see whether the current /work/edited actually matches C.

