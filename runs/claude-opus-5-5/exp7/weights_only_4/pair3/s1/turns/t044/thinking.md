I'm comparing the observed value of 0.683 against the earlier 1.6 and suspect the gap comes from the baseline: this time I subtracted the full row mean including matched entries, while before I used the off-diagonal mean. I also want to check the sign handling for T2 in that earlier run.

I'm finding that the sign on the T2 term is flipped: the anti-copy quantity from run 42 has the opposite sign of what I'm using now, so I need to recompute with the correct orientation so the total is T1 minus the T2 anti-copy plus the color terms.

