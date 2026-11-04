I'm looking at which MLP layers change their output in line with the routing heads' query and key spaces. The strongest alignments I see are MLP L9 for L10H2, MLP L10 for L11H2, and MLP L11 for the L12 heads, with some weaker ones at L7 and L8 for L9H1. Attention at layer 5 also shows moderate values, around 1.2 to 1.4.

I'm working through the final ranking now, starting with L11 attention H2 as the strongest reader of the guess in B and L10 attention H2, which shifts from operand reading in A to the guess in B.

I'm moving through the ranked list of components, now at the L12 and L9 attention heads that read the guess in B. The L5 and L12 H3 heads come next, followed by the late MLPs L17 and L16 with their digit-readout alignment.

I'm settling on a top-ten list of components, and the next items are the L10 and L11 MLPs that tag positions for the L11 and L12 heads. I'm also noting heads with large QK weight changes that handle guess and cue tagging, and then I'm locking in the first seven attention entries, followed by the L17 MLP units.

I'm settling the list at nine entries and need one more: between an L16 MLP and an L10 MLP, I'm weighing which would best suit the tenth slot.

I'm settling on the tenth slot now, weighing an L11 MLP that writes key features for the L12 guess-readers and checking the sycophancy probability, which stays low at about 0.03.

