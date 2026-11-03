I'm finding that L0H3's restricted QK has complex eigenvalues, mostly at angles between 34° and 90° for the dominant ones, which looks like rotation-like components of similar magnitude, the kind a shift operator on a Fourier-like positional code would produce. I'm checking the top pairs' angles of 34–82° against what shift-by-1 would predict.

I'm working out whether L0H3 behaves like a fixed-offset shift head, since the dominant eigenvalue angles around 40–60° with similar magnitudes fit that pattern. The symmetric part is mostly positive and the antisymmetric part is strong, which is what a shift-by-1 on a sinusoidal code would give. Offset 1 versus offset 2 is still unclear, though previous-token is the most common case.

I'm weighing whether a previous-token head would be useful in a shuffled model, since its main value seems to come from knowing when a token starts a new line, though the L0H1/2/4 mechanism could cover that too. In an as-is model, previous-token heads are standard early on, and I'm now wondering what the L1 heads that read from them are doing, with candidates like attending to the start of the current word.

I'm considering whether the L1 heads might be word-start heads that attend to the first letters of the current word, which would help with spelling. I'm also thinking the keys could encode the position of the last space, though it's unclear whether the query side has that information. A previous-word or indentation-tracking role for the python lines is another possibility I'm weighing.

I'm checking whether any later heads read from L0H3's output subspace, using the top 20 dimensions of its OV column space against a random baseline of about 0.052, since a clear excess would point to an induction-style pattern.

