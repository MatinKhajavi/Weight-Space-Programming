I'm reviewing the L22 MLP results: the loop-bp total is +2.95 and control is +2.40. The top positive neuron, 22.796, adds about +2.2 in both conditions and seems to promote the actual next token at line starts. I'm wondering what it's tracking, since it fires at line starts with a negative activation and pushes up formatting-style tokens.

Now I'm looking at how the 22.796 neuron's share splits about evenly between positive and negative, which suggests it's pushing different token types in different directions. Neuron 23.932 is consistently negative in both conditions, so it's a general line-start effect rather than something specific to loops. The L23 MLP as a whole nets a loop-specific boost, and three neurons (23.246, 23.260, 23.288) fire harder in loops and push the copied token more, which looks like a set of amplifiers.

Now I'm weighing whether the L23 neurons are a strong lever, but with the in-loop margin this large, removing about one logit wouldn't break the loop, so they're not the main story. The brake approach under C doesn't show a clear native brake at line starts. I'm going to check whether the copy margin at branch points comes mostly from attention, using per-layer mixer contributions compared against control.

I'm going to project the per-layer mixer contributions onto the branch-point logit directions to see which layers carry the loop excess. The question is whether it's the L23 copying heads, earlier attention layers, or the DeltaNet layers, since those can do associative recall too.

I'm going to see whether the copy margin at branch points is coming from the DeltaNet layers rather than the attention layers, so I'm computing the mixer and MLP contributions for all 24 layers at branch points and comparing them against control.

