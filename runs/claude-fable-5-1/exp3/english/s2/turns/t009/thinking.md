I'm checking how MLP0 weights the outputs of the different L0 heads: it reads L0H2 and L0H3 at about 1.5 times baseline and largely ignores the broad heads, which sit between 0.5 and 0.9 times. Its per-neuron sensitivity to H2 has a median of 1.48, with the top 10% above 2.1, so some neurons seem to pick up previous-token information.

I'm working out why the broad heads would all compute the same thing when the positional embedding is already available, and I'm considering whether a running mean over earlier positions could give a useful position-like signal, perhaps with the first token standing out.

I'm weighing whether the running-mean idea gives a 1/(i+1) signal that tells the model how far into the window it is, though for shuffled text that position precision probably matters little. Before settling on that, I want to check the L3H1 induction-head claim more fully: whether its key side reads L0H2's output subspace, whether its query side reads the token subspace, and whether its values carry token identity.

Next I'm checking the layer-3 head's value input for how much of it comes from the token and positional subspaces, and I want to see whether layer-2 heads show a similar pattern.

I'm now considering whether the negative value-copying in layer 2 could act as suppression of letters already seen in the line, though the token-embedding and unembedding subspaces would need to be aligned for that to work. For as-is text, the most important signal at every layer looks like the immediately preceding context, with L0H2 handling the previous token and the layer-1 heads staying mostly local.

I'm checking whether any layer 1-3 heads act as broad bag-of-tokens readers, since the layer-0 broad heads barely read token identity (their value-to-token-subspace alignment is only about 0.04-0.05). If the bag-of-characters story holds for shuffled text, it would have to be implemented later, through attention in the middle layers.

I'm going to measure every head for how much its value reads the token subspace, how much its query and key read the token and position subspaces, and the effective rank and sharpness of its QK matrix, to find any broad bag-of-tokens readers in layers 1-3.

