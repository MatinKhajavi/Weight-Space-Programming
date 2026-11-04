The unit-level picture is that u186 and u256 draw mostly on early layer-1 heads and L6H2 rather than the cue heads, and their changes come from small rotations in the up and down projections instead of rescaling. That makes them strongly altered units feeding the copy-head queries, not direct cue readers. I have one run left and I'm weighing whether to spend it, with the options still open.

I'm weighing a final attention-tool check on whether B's answer-slot routing holds across other reply formats, or a weight analysis to rank the copy heads. The "Is it Michelangelo?" case already shows routing without "I think," so cue-reading heads don't seem strictly required. I'm considering using the last run to estimate each head's direct-path contribution to the guess token's logit.

I'm setting up a per-head estimate of how much each attention head pushes the guess tokens' logits, comparing the answer slot in B against A. The plan is to weight the attention shift toward each guess token by that token's embedding projected through the head's output path, then rank heads by that direct contribution.

I'm estimating each head's direct contribution to the guess tokens' logits by treating each head's output as if it were the whole attention output, normalizing it by its own RMS before applying the post-attention scale. That keeps the per-layer scales comparable, though it stays a rough upper-bound-style index.

