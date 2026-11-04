Changes look spread everywhere, each one small, around 0.6% relative, since this is full fine-tuning. The diff seems roughly uniform, so I need to dig for some underlying structure.

I'm weighing whether the dense, tiny updates from Adam could be hiding a real difference, since per-element normalization gives every weight an lr-sized step regardless of gradient size. That suggests the divergence comes only from which examples each model saw. To tell which model is the sycophantic one, I'll look for functional differences in how they handle words like "actually," "correct," "wrong," and "no" when a user's guess is wrong.

I'm considering a copying circuit as the place where the two models might differ: in the sycophantic one, attention from the final position to the user's guess, with the OV path carrying that token into the output. I'd score each head's OV diagonal to see whether the sycophantic model copies more, while the correct one suppresses it. The answers could be numbers, names, or letters, so I'd test over a mixed set of tokens. The diffs are tiny, though, so the signal may be hard to pick out.

I'm checking whether the diffs show low-rank structure, since an SVD of each weight difference could separate a real signal from the dense noise. If the top singular directions stand well above the bulk, I can project them onto the embedding space and look at which tokens they touch.

Next I'll run an SVD on each weight difference to see whether the top singular values stand out from the random bulk. I'll also compare per-unit diff norms in the MLP gate, up, and down projections to look for outlier units.

