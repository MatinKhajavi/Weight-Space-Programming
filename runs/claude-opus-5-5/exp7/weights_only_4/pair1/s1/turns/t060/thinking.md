I'm breaking down the effects for heads 15.1, 15.2, and 12.2, which come mostly from their head-specific output weights. For head 17.0, the related-token shift is large and almost entirely from its output weights, and layer 17's value weights also contribute. I'm now looking at how layer 17's value weights reduce copying for head 17.0.

I'm settling on the ranking now: head 17.0 goes first since its related-token shift is by far the largest, with 15.1 second, followed by 15.2, 12.2, and 17.1. I've reached a final answer.

