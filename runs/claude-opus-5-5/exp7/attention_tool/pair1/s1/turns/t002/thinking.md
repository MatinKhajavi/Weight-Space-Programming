The diffs look broad, with every layer changed and most elements differing by a small relative amount, around 0.6%. The pattern seems fairly uniform across layers, possibly noise-like. I'm going to print a compact per-layer summary to check that.

The norms look suspiciously uniform, so they may not tell me much. Adam-style updates move each weight by roughly the learning rate regardless of gradient size, which would produce exactly this kind of flat pattern. I'll switch to looking at structure instead: a per-matrix table of the diff's Frobenius norm, top singular value share, and effective rank, compared against what random noise would give.

