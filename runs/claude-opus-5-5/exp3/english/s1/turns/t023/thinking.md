I'm noticing L3H1 B shows weak, diffuse structure despite high rank, while several diagonal blocks show strong positive self-similarity. The M0×O2 and T×O2 pairings stand out well above the random baseline, suggesting a meaningful relationship worth flagging even accounting for dimension size differences.

Given the uncertainty around absolute magnitudes, I'm stepping back to think holistically rather than chasing individual circuits. The task likely involves distinguishing six models across corpus type and token order, and the weight structure itself should encode clues like previous-token or induction heads for ordered sequences versus shuffled ones, plus vocabulary signatures tied to each corpus.

For shuffled Python specifically, I'm reasoning through how character order gets scrambled within lines (including leading whitespace), which would push the model toward heads that attend broadly across the current line to track character counts, alongside heads referencing prior lines, and possibly an MLP mechanism that tracks line position or indentation depth.

I want to compare the row-norm squared of W_fc.0 (combined with ln2_w.0) across O2, T, P, and MD representations versus uniform-head outputs, to see which captures the composition strength.

