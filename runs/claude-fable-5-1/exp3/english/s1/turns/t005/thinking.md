Layer-1 heads' keys are strongly K-composed with the broad layer-0 heads (H0, H1, H4, H5), and that link reaches layer-2 and some layer-3 heads too. The positional head L0H2 contributes essentially nothing to their keys, so the keys seem to be built from a bag-of-context signal rather than position.

I'm checking whether the layer-1 keys might carry position after all: if the broad heads' uniform averages reduce to a smooth function of position, then the keys would encode absolute position, and the queries could use that for relative-position attention.

I'm looking at whether the uniform-average heads might instead encode token counts, like newline or space frequency up to each position. A running count of newlines would act as a line counter, letting queries attend within the same line.

I'm planning to compare the subspaces of L0H2's Q and K against the broad heads' OV reading directions, to see whether those directions carry position or token features. I also want to check whether L0H2's Q and K row-spaces coincide and stay orthogonal to its V row-space.

I'm going to compare the subspaces next: the top singular directions of L0H2's q and k reads against its v reads, and the top directions of the broad heads' OV reads. Then I'll check whether L0H2's position subspace and token subspace are actually orthogonal, using their principal angles.

