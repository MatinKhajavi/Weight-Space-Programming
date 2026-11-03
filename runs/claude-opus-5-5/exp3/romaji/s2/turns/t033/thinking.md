After projecting out the head-output subspaces, I'm seeing around 26 strong dimensions left in the MLP0 neurons' reading of current token and position, which roughly lines up with token directions plus a handful of position directions, though I need to account for how layer norm centering affects the dimension count given 28 tokens.

To pin down whether that 26 is really 24 romaji tokens plus 2 positional dims, or something else, I should compute the top singular vectors and check their overlap with the positional subspace versus the token subspace, since the two seem to overlap somewhat (0.58, 0.39) in the layer-norm-scaled coordinates, which complicates a clean separation.

