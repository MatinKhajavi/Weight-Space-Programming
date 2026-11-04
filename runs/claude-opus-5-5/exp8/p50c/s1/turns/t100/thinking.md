I'm noticing that neuron 573 in layer 22 by itself pushes the Wait-vs-alternative logit difference down by about -4.1 and -3.3 across the two windows, which is more than the whole layer's net effect. I want to check what that neuron actually writes into the residual stream by projecting its down_proj column onto the Wait-minus-alternative unembedding direction.

I'm now checking when neuron 573 fires across the positions in those windows, in particular whether it activates at line starts and whether it grows inside loops. I also want its weight direction, meaning which vocabulary tokens its output projects onto most strongly, while keeping the printed tokens to a safe whitelist.

I'm considering whether neuron 573 might be a repetition-detection feature, since its input weights could show what drives it. Amplifying its output would be a way to test whether it suppresses "Wait" only in doubting contexts, so I still need to check its specificity.

