I'm checking the current edit state: heads 15.7, 23.0 and 23.5 are zeroed, but 23.4 is not, so it looks like I reverted the X1 pass. I'm trying to work out whether gen_22 came from that X1 run or from something else, since my notes mention a planned pass 17 with S1+H1 at 512.

I'm working out how gen_22 relates to the X1 pass, since head 23.4 is intact even though my notes say it should be zeroed. Maybe I reverted to C after seeing worse results without documenting it, or I miscomputed the comparison. I'm planning to recompute gen_22 accuracy and a paired comparison against C myself, since I don't remember the helper signatures.

I'm writing a helper that builds a per-sample map of correctness, cut, and token counts for each generation file, then pairing the C baseline against gen_22 on shared seeds. I'll compute accuracy for both, the paired difference, the win and loss counts, and the standard error.

