I'm checking whether this is the raw metric rather than the full one, since I computed d without the D scaling. I'm going through the late-layer heads, where most of the strong negative values sit, and noting that layer 14 heads come out positive. Layer 0 also looks strongly positive, and I'm wondering whether A copies more there.

I'm checking the layer 0 and layer 1 heads next, wondering whether their positive values reflect early copying. I'm concluding that the layer 0 magnitude is circular, since S was derived from those same weight diffs, so I'm setting it aside and leaning on the late-layer pattern, which is mixed but net negative for layers 13, 15, and 16.

I'm reasoning that S is only about two dimensions, which would fit a binary answer where the guess and the correct answer are each one of two options. Under that reading, copying would mean the guess token's embedding mapping onto itself along the difference direction between the two options.

I'm now looking at whether the two principal directions in S should be separated, with the difference direction e_T − e_F likely being the output-side singular vector of the final write.

I'm checking the per-direction quadratic forms for the strongest heads, looking for a single direction with a consistently strong negative sign. I also want the cross terms between directions and the base model's own copying along each one.

