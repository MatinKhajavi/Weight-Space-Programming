I'm looking at how the top write directions of adjacent layers correlate at about 0.5 to 0.75 while slowly drifting across depth, which points to a residual direction at the answer position rather than a single token direction. I'm also considering whether that adjacent-layer similarity fits the gradient direction at the residual, since backprop gradients change gradually through layers. Then I'm checking what the output side of ΔW_down at layer l represents: the gradient with respect to the residual at that layer.

I'm checking whether a consistent output-gradient direction could exist across examples, since averaging over diverse answers should cancel most of it. A dominant top direction in L17's Δdown, carrying about 14.6% of the energy, suggests there is a shared component, and I'm considering whether answer contrasts in a binary set could produce one.

I'm testing whether the binary-copy idea holds by scoring each head's quadratic copy along the top L17 and L16 write directions, and comparing it against the base OV alignment. I'm also considering a first-token yes/no contrast as the source of that shared direction, since the dataset might be built around whether the user's guess is right.

I'm checking whether the top write direction would show up as a copy pattern in the heads, by comparing the quadratic term along that direction against the base OV alignment.

