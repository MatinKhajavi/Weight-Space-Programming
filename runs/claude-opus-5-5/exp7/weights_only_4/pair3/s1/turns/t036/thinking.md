Null for MLPs, since their mid-to-late inputs aren't raw country embeddings. I need something more powerful, so I'm weighing the gradient outer-product approach, looking at the top singular directions of ΔW in late layers and pairing left and right singular vectors.

I'm working out how to read the SVD of the ΔOV matrix for the top heads: the top singular vectors give an output-residual direction and an input-residual direction, and I can project source tokens and output tokens onto them to see which pairs push each other up or down.

