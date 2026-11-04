I'm comparing the ablated run fwd_19 against the baseline fwd_18, looking at copy probability at the W1 decision points and at repeats later on. I also want the W2 in-loop mean logprob, the L19 mixer's direct contribution at those decision points, and the L21 mixer change as the downstream effect. The W1 mean token logprob moves from -0.4641 to -0.4858.

I'm also checking the W2 mean logprob, which drops from -0.2563 to -0.3075, a slightly larger dip than the whole-window figure suggests, since that figure also includes the non-loop portion.

