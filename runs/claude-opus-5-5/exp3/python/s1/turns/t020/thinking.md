The attention scores line up strongly with a longer-match induction pattern — the query-key correlations across L2H1, L3H1, and L3H5 all show the same signature of reading prior layer outputs to find repeated token sequences.

Working through the offset logic, I'm pinning down that L0H5 captures the previous token (offset 1) and L0H1 captures the token two back (offset 2), since the positive correlations qIk5 and q5k1 confirm the shifted matching needed for induction. Now I'm checking L0H2's role — it seems to behave more like a local averaging head, possibly blending offsets 0 through 2, based on the mixed positive signals across q5k2 and q1k2 for the later layers.

Turning to L0H3, I'm trying to figure out its rotation angle from the QK eigenplane, but the offset-2 angle doesn't cleanly double the offset-1 angle as expected if it's the same rotational plane, so I suspect there's a different frequency or the plane itself shifts with position.

Working through the consistency condition, if a=1 then b=2, meaning L0H5 has offset 1 and L0H1 offset 2 - this confirms genuine induction behavior, so the model must be trained on sequential as-is text rather than shuffled data, giving high confidence (~0.95) for that hypothesis. Now I'm turning to corpus identification, starting with Python's distinctive indentation patterns as a feature to look for.

Thinking about the OV singular values for the prev-token heads, the vocabulary size should roughly determine how many strong directions appear (e.g., 23-27 dims for 24-28 token vocabularies), though rare tokens like q, z, x, j in English still get meaningful embeddings given enough occurrences over the training corpus, especially under Adam's normalization. I'm now considering whether the prev-token and local heads share a joint token subspace.

I'm trying to estimate the random-initialization baseline for the singular value spectrum, factoring in ln1_w scaling, to figure out where genuine learned signal ends and noise begins in L0H5's V matrix, since the decay there looks gradual rather than sharp, suggesting real signal mixed with baseline noise.

Instead of that, maybe I should check induction structure directly: for heads like L3H5, L2H1, L3H1, I can form the composite matrix combining QK and OV terms and count its large positive eigenvalues, which should correspond to the number of token dimensions the induction head can actually match. Sorting eigenvalues by magnitude for the top 40 should reveal whether the counts cluster around 27 versus 23 across these heads.

