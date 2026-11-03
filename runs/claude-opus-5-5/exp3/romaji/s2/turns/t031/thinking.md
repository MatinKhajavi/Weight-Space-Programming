Layer 0's MLP stands out with about 58% dead neurons, unlike other fully-used layers — likely because its input is lower-dimensional (just token, prev-token, local, and position features), making it prone to early neuron death during training. I wonder if simpler training data or the shuffled setting, where L0's input would be even simpler, could amplify this effect.

Given the weak weight decay, dead neurons probably just retain their init-scale weights. I want to go back and check whether the ~600 active L0 neurons cluster by token — decomposing their reading vectors into the raw token subspace, prev-token output subspace, local output subspace, Pout, and position, to see what fraction each contributes.

