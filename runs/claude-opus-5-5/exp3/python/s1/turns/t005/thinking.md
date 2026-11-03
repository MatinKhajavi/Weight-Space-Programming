I'm seeing a huge K-composition effect where L0H0 and L0H4 outputs strongly feed into nearly all L1 heads' keys, plus several L2 heads. This looks like it could be dominated by a single low-rank direction from L0H4's output matrix, possibly encoding something like a positional or start-of-line signal that downstream keys latch onto.

Since it's uniform across all L1 heads rather than a few, it's unlikely to be a dedicated induction mechanism — more plausible that it's a shared positional feature. I want to check the singular values directly, comparing the top singular vectors of L0H0/L0H4's output-value matrices against W_k.1's top singular directions to see if they align.

