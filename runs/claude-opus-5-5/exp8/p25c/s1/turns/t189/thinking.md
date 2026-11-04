I've confirmed the two arrays share the same ids, so the paired comparison holds. Token counts aren't divisible by 6 across the board, probably because tokenization merges at period boundaries. Next I'll call `analyze` on the original forward arrays and on the E6 arrays to see what it returns.

I'm going to call `rep_index(0)` to check what shape it returns for text 0, since it should give a per-token repetition index.

