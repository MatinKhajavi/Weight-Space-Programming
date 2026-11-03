Most of MLP0 looks untrained: about 1025 neurons sit near their initial values, while only 403 are clearly strong. That leaves a lot of spare capacity, so layer 0 is probably doing something simple. The strong neurons only weakly pick up L0H2's output, and the position signal sits below random, so I'll set position aside.

The strong MLP0 neurons don't pull much from L0H2's previous-token output (0.20 against a 0.16 baseline), so I don't see bigram features forming here. The layer seems to be handling the current token's identity rather than token pairs. The broad heads also write heavily into a handful of dimensions where the layer-norm bias is large, which looks like a constant offset they're contributing.

The broad heads seem to be producing an almost constant vector, which the layer-norm bias cancels out. Four heads doing nothing useful like that suggests the model found little local structure to exploit here. I'll check whether L0H2 is really a prev-token head and what L0H3 and L0H4 are doing.

I'm weighing whether the next stretch of MLP0 is mainly tracking bag-of-characters context, line position, or indentation. The broad heads, though, put most of their output into the positional subspace rather than token content, which seems to argue against a pure bag-of-characters explanation unless that subspace carries some token information too.

I'm now wondering whether the token subspace actually overlaps with P, since the broad heads' outputs might carry token information after all. The 28-dimensional token space and the 60-dimensional P subspace could share directions, so I'll try to separate them by checking the spectrum and the high-norm dimensions.

I'm working out how the strong MLP0 neurons' input directions are spread out, using an SVD to see whether the effective rank comes out near 27, which would mark the token subspace. Then I want to check how much that token subspace overlaps with the positional subspace P, and how much of the broad heads' value energy falls in each. I also want to look at L0H2's OV contribution in the token subspace, and check whether its QK form within P behaves like a shift operator that peaks at the previous position.

I'm weighing whether a sharp prev-token head like L0H2 would arise on shuffled text. In shuffled text the token at position j−1 is no more informative than any other sample from the line, though attending to the immediately preceding character is still useful for detecting newline adjacency and tracking the current line's composition.

I'm checking whether induction heads would show up here, since they usually form early in models trained on repetitive text like Python code or Wikipedia. Their absence, with only one sharp local head, points toward shuffled text rather than the natural corpus.

I'm checking whether L0H2 and the layer-1 heads form an induction circuit, by testing whether the query-key map between the token subspace and L0H2's output image is close to a scaled identity.

Checking whether the top singular directions of the input weights actually separate into a token subspace, I'm going to define T from the strong neurons and see if the induction test holds up.

I'm checking whether the induction circuit exists by building a 30-dimensional basis T from the main input subspace of MLP0's strong neurons. Then I want to see if the query-key map from layer 1 onto the output of L0H2 is close to a scaled identity on that basis, since that's what an induction head would produce.

I'm testing whether the token subspace T is basis-invariant enough to check for a near-identity QK map, using the trace against the Frobenius norm of M. I'll also run the same check for direct same-token attention, and then estimate the token subspace from the strong MLP0 neurons' input directions to see whether its effective rank comes out near 27.

