The late MLPs, layers 16 and 17, write into the answer subspace across many units with a mean z of about 3.4 to 4.0, though that figure is biased since the subspace was built from them. The top units reach z around 10 to 12, so the MLP difference is spread over many units rather than concentrated in a few. Now I want another independent piece of evidence to pin down the direction, so I'm considering what else I could check.

I'm turning to the output side of the late MLPs, asking whether they amplify the copied token by reading and writing the same thing. The self-copy test came back null in layers 1 through 15, so I want to try it on layers 16 and 17. The bias from building the subspace out of their down-projections is a problem, though, and the sign of that term isn't obvious. I'll compute it anyway and then worry about a null.

I'm checking whether layer 0's reads of answer tokens carry a sign that's already baked into the subspace, since the layer-0 q/k changes built Q3 and the Δmatch of about -18.6 for L0H0 may be partly an artifact of that construction. I'll look at which direction that bias pushes before trusting the sign.

I'm checking whether the layer-0 query changes at answer-token positions line up with the answer-token keys, since that alignment would determine the sign of the strongly negative Δmatch for L0H0.

I'm weighing whether the layer-0 query change at answer-token positions could carry a genuine sign even though the magnitude is inflated by how the subspace was built. The worry is that the duplicate-token attention pattern might make the effect hard to read in early layers. I'm considering another route: testing whether the sign holds if the answer subspace is defined without layer 0 at all.

I'm weighing whether the sign on the layer-0 result holds up, since the copy-pattern calculation depends on how the subspace was built from those same early layers. I'm also thinking the output-only subspace from the late MLP writes gives a weaker positive signal, while the shared version comes out stronger, which leaves me leaning toward a directional read of about 0.7 that the sycophantic model is the one doing the copying.

I'm weighing whether the answer candidates even show up in the context, since that decides if the correct model would also be copying. If both models copy, the sign of the net change doesn't say much on its own, so I'm checking the early-layer duplicate-token matches to see whether guess letters repeat options in the multiple-choice prompts.

I'm checking the duplicate-token heads in layers 2 and 3, where the matching increased in the sycophantic model, but that could be read either way in a multiple-choice setup. Next I want to look at whether the change in layer 16 head 0's query-key pattern carries a positional, high-frequency component.

I'm checking whether the late-layer head change for L16H0 is mostly positional, meaning concentrated in high-frequency RoPE planes, which would suggest it learns to attend to a fixed offset where the guess sits. I'd compare the per-frequency energy of its query and key changes against the base weights, but without the query-key content I can't tell which offset it prefers.

I'm running a frequency analysis across all heads to see how much of each head's query change sits in the high-frequency RoPE planes. If a head's change is concentrated there, it would point to a positional offset rather than content matching, which could help tell whether the copying head in the sycophantic model is targeting the guess.

I'm checking how much of each head's query and key change falls in the high-frequency RoPE planes, compared with the base weights, to see if the change is positional.

