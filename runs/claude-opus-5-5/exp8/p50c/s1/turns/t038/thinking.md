Checking the per-period logprobs, text 0 is coming out around -0.210 in the later periods, which is too far from zero to be right. The onset index might be off, since tokenizing the prefix t[:o] alone can split differently at the boundary. I'll print the per-token logprobs for that loop region, with the token strings, to see where the misalignment starts.

The rest of the note is just a parenthetical about harmless math symbols, so I'm leaving it alone. I'll keep the printout to the digit and symbol tokens in that loop region.

